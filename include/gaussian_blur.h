#ifndef GAUSSIAN_BLUR_H
#define GAUSSIAN_BLUR_H

namespace rgf {

// ガウシアンブラーの実行
// input: 入力画像データ（float配列）
// output: 出力画像データ（float配列）
// width: 画像の幅
// height: 画像の高さ
// channels: チャンネル数（1=グレースケール、3=RGB等）
// sigma: ガウシアンブラーの標準偏差
void gaussian_blur_cuda(const unsigned char* input, unsigned char* output, int width, int height, int channels, float sigma);

} // namespace rgf

#endif // GAUSSIAN_BLUR_H 