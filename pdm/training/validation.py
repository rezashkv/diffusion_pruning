import torch
import torch.nn.functional as F
from pdm.utils import compute_snr
from tqdm.auto import tqdm


def validation(dataloader, hyper_net, quantizer, unet, vae, text_encoder, noise_scheduler, config, accelerator,
               global_step, weight_dtype):
    progress_bar = tqdm(
        range(0, len(dataloader)),
        initial=0,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    val_loss = 0.0
    for step, batch in enumerate(dataloader):

        hyper_net.eval()
        quantizer.eval()
        unet.eval()
        unet.reset_flops_count()
        with accelerator.split_between_processes(batch) as batch:
            batch = {k: v.to(accelerator.device) for k, v in batch.items()}
            with torch.no_grad():
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if config.model.unet.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += config.model.unet.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )
                if config.model.unet.input_perturbation:
                    new_noise = noise + config.model.unet.input_perturbation * torch.randn_like(noise)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if config.model.unet.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(latents, new_noise, timesteps)
                else:
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                arch_vector = hyper_net(encoder_hidden_states)
                arch_vector_quantized, q_loss, _ = quantizer(arch_vector)

                unet.set_structure(arch_vector_quantized)

                # Get the target for loss depending on the prediction type
                if config.model.unet.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(prediction_type=config.model.unet.prediction_type)

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                if config.training.losses.diffusion_loss.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                            torch.stack(
                                [snr, config.training.losses.diffusion_loss.snr_gamma * torch.ones_like(timesteps)],
                                dim=1).min(dim=1)[0] / snr
                    )

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(config.data.dataloader.validation_batch_size)).mean()
                val_loss += avg_loss.item()

                progress_bar.update(1)
                step += 1
            accelerator.log({"val_loss": val_loss}, step=global_step)
            logs = {"step_loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)
