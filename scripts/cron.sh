echo 'Syncing summaries and checkpoints from lrzserver ' "$1"

rsync -Pav -e "ssh -i $2" ubuntu@"$1":/ssdtemp/checkpoints  "$3"
rsync -Pav -e "ssh -i $2" ubuntu@"$1":/ssdtemp/summaries  "$3"