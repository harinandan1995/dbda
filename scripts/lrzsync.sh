ssh -i "$2" ubuntu@"$1" "cd ~ && rm -rf dbda && git clone https://ge25gog:jBBMUPdAMxGayy5dEEYt@gitlab.lrz.de/ge25gog/building-design-assistant.git dbda"
ssh -i "$2" ubuntu@"$1" "bash ~/dbda/scripts/lrzconda.sh"

croncmd="bash $3/scripts/cron.sh $1 $2 $4> $4/out.log"
cronjob="*/1 * * * * $croncmd"
( crontab -l | grep -v -F "cron.sh" ; echo "$cronjob" ) | crontab -


# Usage
# 1st argument - ip of the lrz server
# 2nd argument - path to the ssh key
# 3rd argument - path to the repo
# 4th argument - path to the folder where the output after training should be synced
# bash lrzsync.sh 10.195.2.3 ~/TUM/sose2019/IDP/lrzsync/my-openssh-key ~/TUM/sose2019/IDP/building-design-assistant ~/TUM/sose2019/IDP/lrzsync
