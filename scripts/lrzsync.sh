ssh -i "$2" ubuntu@"$1" 'mkdir -p ~/out'

rsync -Pav -e "ssh -i $2" --exclude=env --exclude=.idea --exclude=out "$3" ubuntu@"$1":~/
rsync -Pav -e "ssh -i $2" "$5" ubuntu@"$1":~/datasets

ssh -i "$2" ubuntu@"$1" "bash ~/building-design-assistant/scripts/lrzconda.sh"

croncmd="bash $3/scripts/cron.sh $1 $2 $4> $4/out.log"
cronjob="*/1 * * * * $croncmd"

( crontab -l | grep -v -F "cron.sh" ; echo "$cronjob" ) | crontab -


# Usage
# 1st argument - ip of the lrz server
# 2nd argument - path to the ssh key
# 3rd argument - path to the repo
# 4th argument - path to the folder where the output after training should be synced
# 5th argument - path to the tfrecords folder
# bash lrzsync.sh 10.195.2.3 ~/TUM/sose2019/IDP/lrzsync/my-openssh-key ~/TUM/sose2019/IDP/building-design-assistant ~/TUM/sose2019/IDP/lrzsync ~/TUM/sose2019/IDP/datasets/tfrecords
