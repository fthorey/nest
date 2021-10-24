# How is peeing on my front door ? 

## Setup 

1. Plug your camera on your machine 
2. Find out its name 
```
sudo apt-get install v4l-utils
v4l2-ctl --list-devices
```
3. Replace `ATTR{name}=="MY_CAMERA_NAME"` in scripts/10-camera.rules
4. Setup the dev rule and start the stack.
```
make install-dev-rules
make start
```

If everything works - you'll have a webapp with livestream / stats / last detection in the browser on `0.0.0.0:8501`.
