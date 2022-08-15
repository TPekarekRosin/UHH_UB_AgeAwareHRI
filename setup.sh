if [[ "$OSTYPE" =~ ^linux ]]; then
	which pip
	if [[ $? != 0 ]]; then
		# install pip
		apt install python3-pip
	fi
  pip install wget
  sudo apt-get install sox libsndfile1 ffmpeg  portaudio19-dev swig
  pip install -r requirements.txt
  pip install numpy==1.21
  pip install setuptools==59.5.0
fi