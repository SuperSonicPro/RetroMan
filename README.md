# RetroMan Render Engine

**RetroMan** is a retro-focused rendering engine for **Blender 5.x and later** designed to recreate the visual character and rendering techniques associated with **mid-1990s RenderMan-era CGI**, while fitting into a modern Blender workflow.

RetroMan is intended for artists who want the look and limitations of high-end 1990s CGI without having to work as though they were actually using a workstation from 1995. Scenes are created normally in Blender using familiar objects, materials, cameras, and lights, while RetroMan handles the retro rendering pipeline.

## What RetroMan Can Do

RetroMan aims to reproduce the major characteristics of a circa-1995 production renderer while taking advantage of modern hardware.

Current and planned capabilities include:

- Render Blender scenes with a **mid-1990s RenderMan-inspired appearance**
- Integrate directly with **Blender 5+**
- Use normal Blender scene construction and lighting workflows
- Generate and use **shadow maps** rather than requiring artists to manually prepare them
- Reproduce period-appropriate shading and lighting behavior
- Support texture-mapped materials and procedural shading
- Provide RetroMan-specific material and rendering controls
- Provide an interactive Blender viewport representation of the RetroMan result
- Support final rendering through Blender's normal **F12** workflow
- Use modern CPU/GPU hardware to perform historically expensive rendering techniques much faster than original 1990s hardware
- Target approximately a **GeForce GTX 1650 or equivalent** as the baseline GPU
- Preserve the aesthetic and useful limitations of the older rendering pipeline without deliberately reproducing its original rendering speed

The objective is **not to recreate the old RenderMan user experience exactly**. RetroMan instead attempts to make that class of rendering accessible through the Blender workflow modern artists already know.

## Current Limitations

RetroMan is currently under development and should **not yet be considered production-ready**.

Known limitations include:

- Final **F12 rendering can currently freeze or fail to proceed** in some configurations.
- Viewport integration is incomplete. In affected builds, the viewport may display Blender's normal material preview rather than a true RetroMan-rendered preview.
- Blender 5+ render-engine integration is still being refined.
- Some RenderMan-era shading behavior is approximated rather than being a bit-for-bit reproduction of Pixar's historical renderer.
- Compatibility and performance have not yet been validated across the full range of supported GPUs, operating systems, and Blender configurations.
- RetroMan is intended to reproduce the **look, techniques, and practical capabilities** of the period rather than provide binary compatibility with historical Pixar RenderMan software.
- Scenes relying heavily on modern Blender/Cycles-specific features may not translate directly to RetroMan's deliberately older rendering model.

Until the F12 and viewport integration paths are fully stabilized and extensively tested, RetroMan should be treated as **experimental software**.

## Pixar / RenderMan Credit

**RenderMan is a technology and trademark of Pixar Animation Studios.**

RetroMan is an independent project and is **not affiliated with, sponsored by, endorsed by, or an official product of Pixar Animation Studios**.

Where Pixar-provided RenderMan demonstration assets, shaders, scenes, models, textures, documentation, or other materials are included or used for compatibility/testing purposes, those materials remain the property of their respective copyright holders and are provided subject to their original licenses and terms.

**Pixar, RenderMan, and associated names and marks are trademarks of Pixar Animation Studios and/or their respective owners.**

RetroMan itself does not claim ownership of Pixar RenderMan assets or other third-party materials.

The project also acknowledges earlier open-source work exploring RenderMan-style rendering, including **JrMan**, an open-source RenderMan-related implementation referenced during RetroMan's development.

---

**RetroMan Render Engine**  
*Modern hardware. Modern Blender. 1995 rendering.*
