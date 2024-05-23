#!/bin/bash
set -e
############# setup user enviroment ##########
export USERNAME=${USERNAME:-docker}
export USERID=${USERID:-50000}
HOMEDIR=/home/${USERNAME}
echo "creating user ${USERNAME} with user id: ${USERID}"
#groupadd -g ${USERID} ${USERNAME}
useradd -rm -d ${HOMEDIR} -s /bin/bash ${USERNAME} -u ${USERID}

cp -r /entry/asset/ssh ${HOMEDIR}/.ssh
chown ${USERNAME}:${USERNAME} -R ${HOMEDIR}/.ssh
# chown ${USERNAME}:${USERNAME} -R /opt/conda
su $USERNAME <<EOSU
#/bin/bash
set -e
export PATH=/opt/conda/bin:\${PATH}
cd ~
############## real executing scripts ##########

python \${AF_PATH}/run_af2_stage.py $@
EOSU
