sudo yum -y install lapack-devel.x86_64 blas-devel.x86_64 
sudo yum -y install freetype-devel
sudo yum -y install libpng-devel.x86_64
sudo yum -y install numpy-f2py
sudo yum -y install gcc-gfortran
sudo yum -y install libjpeg-turbo-devel.x86_64
sudo yum -y install git
sudo yum -y install python-pip
sudo pip -y install numpy
sudo pip install scipy
sudo pip install maptlotlib
sudo pip install scikit-learn
sudo pip install scikit-image
sudo pip install jupyter
sudo pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
sudo pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip

curl -X GET -o RPM-GPG-KEY-lambda-epll https://lambda-linux.io/RPM-GPG-KEY-lambda-epll

sudo rpm --import RPM-GPG-KEY-lambda-epll

curl -X GET -o epll-release-2016.03-1.1.ll1.noarch.rpm https://lambda-linux.io/epll-release-2016.03-1.1.ll1.noarch.rpm

# Install firefox
sudo yum -y install epll-release-2016.03-1.1.ll1.noarch.rpm

sudo yum --enablerepo=epll install firefox-compat

wget -O firefox-latest.tar.bz2 \
  "https://download.mozilla.org/?product=firefox-latest&os=linux64&lang=en-US"

wget -O firefox-latest.tar.bz2 \
  "https://download.mozilla.org/?product=firefox-latest&os=linux64&lang=en-US"


bzcat firefox-latest.tar.bz2 | tar xvf -

