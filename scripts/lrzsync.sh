ssh -i my-openssh-key ubuntu@"$1" 'mkdir -p /ssdtemp/datasets'
ssh -i my-openssh-key ubuntu@"$1" 'mkdir -p /ssdtemp/checkpoints'
ssh -i my-openssh-key ubuntu@"$1" 'mkdir -p /ssdtemp/summaries'

rsync -Pav -e 'ssh -i my-openssh-key' --exclude=env --exclude=.idea --exclude=summaries --exclude=checkpoints /home/harikatam/TUM/sose2019/IDP/building-design-assistant ubuntu@"$1":/home/ubuntu
rsync -Pav -e 'ssh -i my-openssh-key' /home/harikatam/TUM/sose2019/IDP/datasets/tfrecords ubuntu@"$1":/ssdtemp/datasets

ssh -i my-openssh-key ubuntu@"$1" 'bash /home/ubuntu/building-design-assistant/scripts/lrzsetup.sh'

croncmd="bash /home/harikatam/TUM/sose2019/IDP/building-design-assistant/scripts/cron.sh $1"
cronjob="0 */1 * * * $croncmd"

( crontab -l | grep -v -F "cron.sh" ; echo "$cronjob" ) | crontab -