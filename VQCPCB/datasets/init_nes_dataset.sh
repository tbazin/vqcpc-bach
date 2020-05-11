ROOT_DIR="$( cd "$(dirname "$0")/../../../data" >/dev/null 2>&1 ; pwd -P )"
ARCHIVE_NAME='nesmdb_midi.tar.gz'

mkdir -p $ROOT_DIR

wget -N http://deepyeti.ucsd.edu/cdonahue/nesmdb/$ARCHIVE_NAME -O $ROOT_DIR/$ARCHIVE_NAME

if tar -C $ROOT_DIR -xf $ROOT_DIR/$ARCHIVE_NAME && rm $ROOT_DIR/$ARCHIVE_NAME; then
  echo -e "\e[32mDataset successfully extracted in $ROOT_DIR.\e[0m"
  exit 0
else
  echo -e "\e[31mSome errors occured during downloading or extraction.\e[0m"
  exit 1
fi
