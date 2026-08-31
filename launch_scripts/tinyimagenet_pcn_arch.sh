set_tinyimagenet_pcn_arch() {
  case "$1" in
    wrn_16_2)
      INP_CHANNELS="3 16 32 32 64 64 128"
      OUT_CHANNELS="16 32 32 64 64 128 128"
      MAX_POOL="0 0 0 1 0 1 0"
      ;;
    wrn_16_4)
      INP_CHANNELS="3 16 64 64 128 128 256"
      OUT_CHANNELS="16 64 64 128 128 256 256"
      MAX_POOL="0 0 0 1 0 1 0"
      ;;
    wrn_16_8)
      INP_CHANNELS="3 16 128 128 256 256 512"
      OUT_CHANNELS="16 128 128 256 256 512 512"
      MAX_POOL="0 0 0 1 0 1 0"
      ;;
    wrn_28_2)
      INP_CHANNELS="3 16 32 32 32 32 64 64 64 64 128 128 128"
      OUT_CHANNELS="16 32 32 32 32 64 64 64 64 128 128 128 128"
      MAX_POOL="0 0 0 0 0 1 0 0 0 1 0 0 0"
      ;;
    wrn_28_4)
      INP_CHANNELS="3 16 64 64 64 64 128 128 128 128 256 256 256"
      OUT_CHANNELS="16 64 64 64 64 128 128 128 128 256 256 256 256"
      MAX_POOL="0 0 0 0 0 1 0 0 0 1 0 0 0"
      ;;
    wrn_40_2)
      INP_CHANNELS="3 16 32 32 32 32 32 32 64 64 64 64 64 64 128 128 128 128 128"
      OUT_CHANNELS="16 32 32 32 32 32 32 64 64 64 64 64 64 128 128 128 128 128 128"
      MAX_POOL="0 0 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 0 0"
      ;;
    *)
      echo "Unsupported MODEL_ARCH: $1" >&2
      return 1
      ;;
  esac
}
