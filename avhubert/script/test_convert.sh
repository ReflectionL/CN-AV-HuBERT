#!/bin/bash
# 测试数据转换脚本（使用少量样本）

echo "Testing CNVSRC data conversion with small subset..."

python convert_cnvsrc_data.py \
    --train-csv /home/lixiaolou/code/CNVSRC2024Baseline/data/cnvsrc-single/train.csv \
    --valid-csv /home/lixiaolou/code/CNVSRC2024Baseline/data/cnvsrc-single/valid.csv \
    --data-root /ssd1/processed/ \
    --output-dir /home/lixiaolou/code/CN-AV-HuBERT/avhubert/bpe/ 
    # --max-train-samples 10 \
    # --max-valid-samples 5

echo ""
echo "Checking generated files..."
echo ""

# 检查生成的文件
for file in train.tsv train.wrd valid.tsv valid.wrd; do
    filepath="/home/lixiaolou/code/CN-AV-HuBERT/avhubert/bpe/$file"
    if [ -f "$filepath" ]; then
        lines=$(wc -l < "$filepath")
        echo "✓ $file: $lines lines"
        echo "  First 3 lines:"
        head -3 "$filepath" | sed 's/^/    /'
        echo ""
    else
        echo "✗ $file not found"
    fi
done
