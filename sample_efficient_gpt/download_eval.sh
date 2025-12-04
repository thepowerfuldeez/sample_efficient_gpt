export SAMPLE_EFFICIENT_GPT_BASE_DIR="$HOME/.cache/sample_efficient_gpt"
rm -rf $SAMPLE_EFFICIENT_GPT_BASE_DIR/eval_bundle
mkdir -p $SAMPLE_EFFICIENT_GPT_BASE_DIR
EVAL_BUNDLE_URL=https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip
if [ ! -d "$SAMPLE_EFFICIENT_GPT_BASE_DIR/eval_bundle" ]; then
    curl -L -o eval_bundle.zip $EVAL_BUNDLE_URL
    unzip -q eval_bundle.zip
    rm eval_bundle.zip
    mv eval_bundle $SAMPLE_EFFICIENT_GPT_BASE_DIR/eval_bundle
fi