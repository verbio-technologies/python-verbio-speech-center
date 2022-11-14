#!/bin/bash 

PYTHON_EXEC=$(which python3)
if [[ ! -e ${PYTHON_EXEC} ]]; then
  echo "[ERROR] python3 must be available";
fi

PROTO_PATH="../proto/"
PROTO_PATH_FILES="../proto/*.proto"
PROTO_PATH_GENERATED="../proto/generated/"

# protoc does not support relative path, so we enter the protoc directory.
for filename in $PROTO_PATH_FILES; do
  PROTO_DIR=$(dirname "${PROTO_PATH}")
  PROTO_FILE=$(basename "${filename}")
  echo "Generating new proto file: $filename PROTO_DIR=${PROTO_DIR} PROTO_FILE=${PROTO_FILE}"
  ${PYTHON_EXEC} -m grpc_tools.protoc -I"${PROTO_PATH}" --python_out=${PROTO_PATH_GENERATED} --grpc_python_out=${PROTO_PATH_GENERATED} "${PROTO_FILE}"
done


