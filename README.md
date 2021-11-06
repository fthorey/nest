[![What cat is peeing on my front door?](https://pimp-my-readme.webapp.io/pimp-my-readme/sliding-text?emojis=1f92f&text=What%2520cat%2520is%2520peeing%2520on%2520my%2520front%2520door%253F)](https://pimp-my-readme.webapp.io)


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
# If you want a summary
make help
# If you want to just get the app
make install-dev-rules
make build
make deploy
make start
# to shutdown
make shutdown
```

If everything works - you'll have a webapp with livestream / stats / last detection in the browser on `0.0.0.0:8501`.
