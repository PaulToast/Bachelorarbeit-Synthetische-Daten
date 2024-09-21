# Convert images to latent space
latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
latents = latents * vae.config.scaling_factor

# Sample noise that we'll add to the latents
noise = torch.randn_like(latents)
bsz = latents.shape[0]
# Sample a random timestep for each image
timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
timesteps = timesteps.long()

# Add noise to the latents according to the noise magnitude at each timestep
# (this is the forward diffusion process)
noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

# Get the text embedding for conditioning
encoder_hidden_states = text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)

# Predict the noise residual
model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

# Get the target for loss depending on the prediction type
if noise_scheduler.config.prediction_type == "epsilon":
    target = noise
elif noise_scheduler.config.prediction_type == "v_prediction":
    target = noise_scheduler.get_velocity(latents, noise, timesteps)
else:
    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

accelerator.backward(loss)

optimizer.step()
lr_scheduler.step()
optimizer.zero_grad()