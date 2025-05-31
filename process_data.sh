#!/bin/bash

QT_QPA_PLATFORM=offscreen ns-process-data video \
    --data /root/3D-Rendering-Contest/CS479-3DML-Project/camera_orbit_multiobject_upscaled.mp4 \
    --output-dir data/final_scene_7 \
    --no-gpu