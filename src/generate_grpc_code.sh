#!/bin/bash 

PYTHON_EXEC=$(which python3)
if [[ ! -e ${PYTHON_EXEC} ]]; then
  echo "[ERROR] python3 must be available";
fi

if [[ $# -ne 3 ]]; then
  if [[ $# -ne 0 ]]; then
    echo "[ERROR] Missing parameters"
  fi
  echo;
  echo "use: $0 <protobuf_definition_file> <python_output_path> <grpc_output_path>";
  echo;
  exit 1;
fi

PROTO_PATH=$1
PYTHON_OUT=$2
GRPC_OUT=$3

PROTO_DIR=$(dirname "${PROTO_PATH}")
PROTO_FILE=$(basename "${PROTO_PATH}")

mkdir -p "${PYTHON_OUT}" "${GRPC_OUT}"

${PYTHON_EXEC} -m grpc_tools.protoc -I"${PROTO_DIR}" --python_out="${PYTHON_OUT}" --grpc_python_out="${GRPC_OUT}" "${PROTO_FILE}"

