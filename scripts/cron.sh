rsync -Pav -e 'ssh -i my-openssh-key' ubuntu@"$1":/home/ubuntu/building-design-assistant/checkpoints  /home/harikatam/TUM/sose2019/IDP/lrzsync/
rsync -Pav -e 'ssh -i my-openssh-key' ubuntu@"$1":/home/ubuntu/building-design-assistant/summaries  /home/harikatam/TUM/sose2019/IDP/lrzsync/