# Vulkan backend

This directory contains the recovered Apple-silicon Vulkan port used by the
M5 Pro Qwen3.8-27B Q6_K qualification.  The port keeps the existing Splash
backend contract (`splash::metal::MetalBackend`) while replacing Metal command
encoding with Vulkan and importing the qualified G17G shader objects through
`VK_EXT_shader_object`.

The source is kept beside the current Metal implementation so the port can be
rebased against the current engine without changing the default macOS build.
The current branch is based on the latest `main`; the October qualification
source was recovered from the retained lab bundle and is recorded in
[`docs/vulkan.md`](../../docs/vulkan.md).

The native binaries in `native/` are driver-qualified artifacts.  They are
optional: an incompatible driver must fall back to the ordinary SPIR-V
compute path rather than importing them.
