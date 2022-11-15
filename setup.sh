if [[ "$OSTYPE" =~ ^linux ]]; then
	which pip
	if [[ $? != 0 ]]; then
		# install pip
		apt install python3-pip
	fi
  pip install wget
  sudo apt-get install sox libsndfile1 ffmpeg  portaudio19-dev swig
  cd age_recognition
  pip install .
  cd ..
fi