ssh -i my-openssh-key ubuntu@$1 'mkdir -p /home/ubuntu/datasets/tfrecords'

rsync -Pav -e 'ssh -i my-openssh-key' /home/harikatam/TUM/sose2019/IDP/building-design-assistant ubuntu@$1:/home/ubuntu
rsync -Pav -e 'ssh -i my-openssh-key' /home/harikatam/TUM/sose2019/IDP/datasets/tfrecords ubuntu@$1:/home/ubuntu

croncmd="bash cron.sh $1"
cronjob="0 */1 * * * $croncmd"

( crontab -l | grep -v -F "bash cron.sh" ; echo "$cronjob" ) | crontab -