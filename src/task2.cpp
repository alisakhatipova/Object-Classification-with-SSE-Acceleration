#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>
#include <tuple>
#include <emmintrin.h>
#include <smmintrin.h>


#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"
#include "matrix.h"
#include "Timer.h"
#include "task2.h"

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;
using std::tuple;
using std::make_tuple;
using std::tie;


using CommandLineProcessing::ArgvParser;
/**
@mainpage My fabulous program
@author This project was created by Alisa Khatipova
*/
/**
@file task2.cpp
*/

/**
@function Get4Pixels16Bit
reads four consecutive pixels of the specified row started from given column and writes they to the
two registers out_BG and out_RA. Uses 16 bit per channel
@param in_img is a input image
@param in_row_idx is an index of a row to read pixels
@param in_col_idx ia an index of a column with a first pixel
@param out_BG is a pointer to a 128bit register to store blue and green channels for the pixels four
consecutive pixels in format BBBB GGGG. Order of pixels is [0, 1, 2, 3]
@param out_RA is a pointer to a 128bit register to store red and alpha channels for the pixels four
consecutive pixels in format RRRR AAAA. Order of pixels is [0, 1, 2, 3]
*/
inline void Get4Pixels16Bit(BMP &in_img, int in_row_idx, int in_col_idx,
                            __m128i *out_BG, __m128i *out_RA)
{
  // read four consecutive pixels
  RGBApixel pixel0 = in_img.GetPixel(in_col_idx, in_row_idx);
  RGBApixel pixel1 = in_img.GetPixel(in_col_idx + 1, in_row_idx);
  RGBApixel pixel2 = in_img.GetPixel(in_col_idx + 2, in_row_idx);
  RGBApixel pixel3 = in_img.GetPixel(in_col_idx + 3, in_row_idx);

  // write two pixel0 and pixel2 to the p02 and other to the p13
  __m128i p02 = _mm_setr_epi32(*(reinterpret_cast<int*>(&pixel0)), *(reinterpret_cast<int*>(&pixel2)), 0, 0);
  __m128i p13 = _mm_setr_epi32(*(reinterpret_cast<int*>(&pixel1)), *(reinterpret_cast<int*>(&pixel3)), 0, 0);

  /*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  * convert BGRA BGRA BGRA BGRA
  * to BBBB GGGG RRRR AAAA
  * order of pixels corresponds to the digits in the name of variables
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
  // BGRA BGRA 0000 0000 + BGRA BGRA 0000 0000 -> BBGG RRAA BBGG RRAA

  __m128i p0123 = _mm_unpacklo_epi8(p02, p13);
  // extract BBGG RRAA 0000 0000 of pixel2 and pixel3
  __m128i p23xx = _mm_unpackhi_epi64(p0123, _mm_setzero_si128());
  // BBGG RRAA XXXX XXXX + BBGG RRAA 0000 0000 -> BBBB GGGG RRRR AAAA
  // X denotes unused characters
  __m128i p0123_8bit = _mm_unpacklo_epi16(p0123, p23xx);

  // extend to 16bit
  *out_BG = _mm_unpacklo_epi8(p0123_8bit, _mm_setzero_si128());
  *out_RA = _mm_unpackhi_epi8(p0123_8bit, _mm_setzero_si128());
}
/*
/// Weight of red channel multiplied by 256. It's stored as 8 equal 16 bit integer values
static const __m128i CONST_RED_16_INT = _mm_set1_epi16(static_cast<short>(0.2125f * 256));
/// Weight of green channel multiplied by 256. It's stored as 8 equal 16 bit integer values
static const __m128i CONST_GREEN_16_INT = _mm_set1_epi16(static_cast<short>(0.7154f * 256));
/// Weight of blue channel multiplied by 256. It's stored as 8 equal 16 bit integer values
static const __m128i CONST_BLUE_16_INT = _mm_set1_epi16(static_cast<short>(0.0721f * 256)); */

/**
@function toGrayScale
utilizes SSE to realize fast approach to convert RGBA image to grayscale.
It is faster than toGrayScaleSSE, but not so precise
@param [in] in_input is an input image.
@param [out] out_mat is an output image. Each pixel is represented by a single unsigned char value.
*/
void toGrayScaleSSE_16BIT(BMP &in_input, MyMat<uchar> &out_mat)
{
  // pointer to the processed row of the result image
  uchar *row_ptr = out_mat.data;
  // pointer to the processed element of the result image
  uchar *elem_ptr;
  // number of elements to process at a time
  const int block_size = 8;
  // number of elements that will not be processed block-wise
  const int left_cols = out_mat.cols % block_size;
  // number of elements that will be processed block-wise
  const int block_cols = out_mat.cols - left_cols;

  for (size_t row_idx = 0; row_idx  < out_mat.rows; ++row_idx)
  {
    elem_ptr = row_ptr;
    // process block_size elements at a time
    for (int col_idx = 0; col_idx < block_cols; col_idx += block_size)
    {
      // read four pixels
      __m128i BG1;
      __m128i RA1;
      Get4Pixels16Bit(in_input, row_idx, col_idx, &BG1, &RA1);

      // read another four pixels
      __m128i BG2;
      __m128i RA2;
      Get4Pixels16Bit(in_input, row_idx, col_idx + 4, &BG2, &RA2);

      // extract channels
      __m128i blue = _mm_unpacklo_epi64(BG1, BG2);
      __m128i green = _mm_unpackhi_epi64(BG1, BG2);
      __m128i red = _mm_unpacklo_epi64(RA1, RA2);

      // multiply channels by weights
      blue = _mm_mullo_epi16(blue, CONST_BLUE_16_INT);
      green = _mm_mullo_epi16(green, CONST_GREEN_16_INT);
      red = _mm_mullo_epi16(red, CONST_RED_16_INT);

      // sum up channels
      __m128i color = _mm_add_epi16(red, green);
      color = _mm_add_epi16(color, blue);

      // divide by 256
      color = _mm_srli_epi16(color, 8);

      // convert to 8bit
      color = _mm_packus_epi16(color, _mm_setzero_si128());

      // write results to the output image
      _mm_storel_epi64(reinterpret_cast<__m128i*>(elem_ptr), color);
      elem_ptr += block_size;
    }
    // process left elements in the row
    for (size_t col_idx = block_cols; col_idx < out_mat.cols; ++col_idx)
    {
      RGBApixel pixel = in_input.GetPixel(col_idx, row_idx);
      short red = 0.2125f * 256, green =  0.7154f * 256, blue =  0.0721f * 256;
      int carry =(red * pixel.Red + green * pixel.Green + blue * pixel.Blue) / 256;
      *elem_ptr = static_cast<uchar>(carry);
      ++elem_ptr;
    }
    // go to next row
    row_ptr += out_mat.step;
  }

}

/** Load list of files and its labels from 'data_file' and stores it in 'file_list'
*/
void LoadFileList(const string& data_file, TFileList* file_list) {
    ifstream stream(data_file.c_str());

    string filename;
    int label;

    int char_idx = data_file.size() - 1;
    for (; char_idx >= 0; --char_idx)
        if (data_file[char_idx] == '/' || data_file[char_idx] == '\\')
            break;
    string data_path = data_file.substr(0,char_idx+1);

    while(!stream.eof() && !stream.fail()) {
        stream >> filename >> label;
        if (filename.size())
            file_list->push_back(make_pair(data_path + filename, label));
    }

    stream.close();
}

/**Load images by list of files 'file_list' and store them in 'data_set'

*/
void LoadImages(const TFileList& file_list, TDataSet* data_set) {
    for (size_t img_idx = 0; img_idx < file_list.size(); ++img_idx) {
            // Create image
        BMP* image = new BMP();
            // Read image from file
        image->ReadFromFile(file_list[img_idx].first.c_str());
            // Add image and it's label to dataset
        data_set->push_back(make_pair(image, file_list[img_idx].second));
    }
}

/** Save result of prediction to file
*/
void SavePredictions(const TFileList& file_list,
                     const TLabels& labels,
                     const string& prediction_file) {
        // Check that list of files and list of labels has equal size
    assert(file_list.size() == labels.size());
        // Open 'prediction_file' for writing
    ofstream stream(prediction_file.c_str());

        // Write file names and labels to stream
    for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx)
        stream << file_list[image_idx].first << " " << labels[image_idx] << endl;
    stream.close();
}

/** get Abs and Direct matrixes without using SSE from BMP image.
 * @param [out] Direct - matrix of directions
 * @param [out] im - input image
 * @param [out] Abs - matrix of gradient absolut values
 */

void get_dir_abs(BMP *im, Matrix<float> &Direct, Matrix<float> &Abs){
    int src_w = im->TellWidth(), src_h = im->TellHeight();
    Matrix<float> Intens(src_h, src_w);
    for (int i = 0; i < src_h; ++i)
        for (int j = 0; j < src_w; ++j){
            RGBApixel pixel = im->GetPixel(j, i);
            short red = 0.2125f * 256, green =  0.7154f * 256, blue =  0.0721f * 256;
            int carry =(red * pixel.Red + green * pixel.Green + blue * pixel.Blue) / 256;
            Intens(i, j) = carry;
    }
        for (int i = 1; i < src_h - 1; ++i)
            for (int j = 1; j < src_w - 1; ++j){
                float y = - Intens(i + 1, j - 1) -  Intens(i + 1, j+ 1) - 2 * Intens(i + 1, j) + 2 * Intens(i - 1, j) + Intens(i - 1, j - 1) +  Intens(i - 1, j+ 1);
                float x = Intens(i + 1, j - 1) + Intens(i - 1, j -  1) + 2 * Intens(i, j - 1) - 2 * Intens(i, j + 1) - Intens(i + 1, j + 1) - Intens(i - 1, j +  1);
                Direct(i - 1, j - 1) = atan2(y, x);
                Abs(i - 1, j - 1) = sqrt(x * x + y * y) * 256 / 1140;

            }
}

/** get Abs and Direct matrixes using SSE from BMP image.
 * @param [out] Direct - matrix of directions
 * @param [out] im - input image
 * @param [out] Abs - matrix of gradient absolut values
 */

void get_dir_abs_SSE(BMP *im, Matrix<float> &Direct, Matrix<float> &Abs){
    uint src_w = im->TellWidth(), src_h = im->TellHeight();
    MyMat<uchar> img(src_h, src_w, 1);
    toGrayScaleSSE_16BIT(*im, img);
    uint h = src_h - 2, w = src_w - 2;
    MyMat<float> Hx(h, w, 1), Hy(h, w, 1), Gr(h, w, 1);
    float *Hx_row_ptr = Hx.data, *Hy_row_ptr = Hy.data, *Gr_row_ptr = Gr.data;
    float *Hx_elem_ptr, *Hy_elem_ptr, *Gr_elem_ptr;
    uchar *Img_ptr = img.data;
   int img_step = img.step;
  const int block_size = 4;
  // number of elements that will not be processed block-wise
  const int left_cols = w % block_size;
  // number of elements that will be processed block-wise
  const int block_cols = w - left_cols;

  for (size_t row_idx = 1; row_idx  < src_h - 1; ++row_idx)
  {
    Hx_elem_ptr = Hx_row_ptr;
    Hy_elem_ptr = Hy_row_ptr;
    Gr_elem_ptr = Gr_row_ptr;
    // process block_size elements at a time
    for (int col_idx = 1; col_idx <1 + block_cols; col_idx += block_size)
    {
      __m128i A = _mm_setr_epi32(*(reinterpret_cast<int*>(Img_ptr + (row_idx - 1) * img_step + (col_idx - 1))), 0, 0, 0);
      __m128i B = _mm_setr_epi32(*(reinterpret_cast<int*>(Img_ptr + (row_idx - 1) * img_step + (col_idx))), 0, 0, 0);
      __m128i C = _mm_setr_epi32(*(reinterpret_cast<int*>(Img_ptr + (row_idx - 1) * img_step + (col_idx + 1))), 0, 0, 0);

      __m128i D = _mm_setr_epi32(*(reinterpret_cast<int*>(Img_ptr + (row_idx) * img_step + (col_idx - 1))), 0, 0, 0);
      __m128i F = _mm_setr_epi32(*(reinterpret_cast<int*>(Img_ptr + (row_idx) * img_step + (col_idx + 1))), 0, 0, 0);

      __m128i G = _mm_setr_epi32(*(reinterpret_cast<int*>(Img_ptr + (row_idx + 1) * img_step + (col_idx - 1))), 0, 0, 0);
      __m128i H = _mm_setr_epi32(*(reinterpret_cast<int*>(Img_ptr + (row_idx + 1) * img_step + (col_idx))), 0, 0, 0);
      __m128i I = _mm_setr_epi32(*(reinterpret_cast<int*>(Img_ptr + (row_idx + 1) * img_step + (col_idx + 1))), 0, 0, 0);

      A = _mm_unpacklo_epi8(A, _mm_setzero_si128());
      B = _mm_unpacklo_epi8(B, _mm_setzero_si128());
      C = _mm_unpacklo_epi8(C, _mm_setzero_si128());
      D = _mm_unpacklo_epi8(D, _mm_setzero_si128());
      F = _mm_unpacklo_epi8(F, _mm_setzero_si128());
      G = _mm_unpacklo_epi8(G, _mm_setzero_si128());
      H = _mm_unpacklo_epi8(H, _mm_setzero_si128());
      I = _mm_unpacklo_epi8(I, _mm_setzero_si128());

      A = _mm_unpacklo_epi8(A, _mm_setzero_si128());
      B = _mm_unpacklo_epi8(B, _mm_setzero_si128());
      C = _mm_unpacklo_epi8(C, _mm_setzero_si128());
      D = _mm_unpacklo_epi8(D, _mm_setzero_si128());
      F = _mm_unpacklo_epi8(F, _mm_setzero_si128());
      G = _mm_unpacklo_epi8(G, _mm_setzero_si128());
      H = _mm_unpacklo_epi8(H, _mm_setzero_si128());
      I = _mm_unpacklo_epi8(I, _mm_setzero_si128());
      __m128 AF = _mm_cvtepi32_ps(A);
      __m128 BF = _mm_cvtepi32_ps(B);
      __m128 CF = _mm_cvtepi32_ps(C);
      __m128 DF = _mm_cvtepi32_ps(D);
      __m128 FF = _mm_cvtepi32_ps(F);
      __m128 GF = _mm_cvtepi32_ps(G);
      __m128 HF = _mm_cvtepi32_ps(H);
      __m128 IF = _mm_cvtepi32_ps(I);
      __m128 carry = _mm_set1_ps(2.0);
      DF = _mm_mul_ps(DF, carry);
      FF = _mm_mul_ps(FF, carry);
      BF = _mm_mul_ps(BF, carry);
      HF = _mm_mul_ps(HF, carry);
      __m128 S1 = _mm_sub_ps(AF, IF);
      __m128 S2 = _mm_sub_ps(CF, GF);
      __m128 H1 = _mm_sub_ps(DF, FF);
      __m128 H2 = _mm_sub_ps(BF, HF);
      H1 =  _mm_add_ps(H1, S1);
      H1 = _mm_sub_ps(H1, S2);
      H2 = _mm_add_ps(H2, S1);
      H2 = _mm_add_ps(H2, S2);
      _mm_storeu_ps(Hx_elem_ptr, H1);
      _mm_storeu_ps(Hy_elem_ptr, H2);
      __m128 Norm =  _mm_set1_ps((1.0f * 256)/1140);
      H1 =_mm_mul_ps(H1, H1);
      H2 = _mm_mul_ps(H2, H2);
      H1 = _mm_add_ps(H1, H2);
      H1 = _mm_sqrt_ps(H1);
      H1 = _mm_mul_ps(H1, Norm);
      _mm_storeu_ps(Gr_elem_ptr, H1);
      Hx_elem_ptr += block_size;
      Hy_elem_ptr += block_size;
      Gr_elem_ptr += block_size;
    }
    for (size_t col_idx =  1 + block_cols; col_idx < img.cols - 1; ++col_idx)
    {
      uchar aa, bb, cc, dd, ff, gg, hh, ii;
      aa = *(Img_ptr + (row_idx - 1) * img_step + (col_idx - 1));
      bb = *(Img_ptr + (row_idx - 1) * img_step + (col_idx));
      cc = *(Img_ptr + (row_idx - 1) * img_step + (col_idx + 1));
      dd = *(Img_ptr + (row_idx) * img_step + (col_idx - 1));
      ff = *(Img_ptr + (row_idx) * img_step + (col_idx + 1));
      gg = *(Img_ptr + (row_idx + 1) * img_step + (col_idx - 1));
      hh = *(Img_ptr + (row_idx + 1) * img_step + (col_idx));
      ii = *(Img_ptr + (row_idx + 1) * img_step + (col_idx + 1));
      float carry1 = aa - cc + 2 * dd - 2 * ff + gg - ii, carry2 =  aa - gg + 2 * bb - 2 * hh + cc - ii;
      *Hx_elem_ptr = carry1;
      *Hy_elem_ptr = carry2;
      *Gr_elem_ptr = (256 * sqrt(carry1 * carry1 + carry2 * carry2))/1140;
      ++Hx_elem_ptr;
      ++Hy_elem_ptr;
      ++Gr_elem_ptr;
    }
    Hx_row_ptr += Hx.step / sizeof(float);
    Hy_row_ptr += Hy.step / sizeof(float);
    Gr_row_ptr += Gr.step / sizeof(float);
  }
     Hx_elem_ptr = Hx.data; Hy_elem_ptr = Hy.data; Gr_elem_ptr = Gr.data;
     for (uint i = 0; i < h; ++i)
        for (uint j = 0; j < w; ++j){
            float x = *(Hx_elem_ptr + i * Hx.step / sizeof(float) + j);
            float y = *(Hy_elem_ptr + i * Hy.step / sizeof(float) + j);
            Direct(i, j) = atan2(y, x);
            Abs(i, j) = *(Gr_elem_ptr + i * Gr.step / sizeof(float) + j);
        }
}

/** get features from matrixes Abs and Direct.
 * @param [in] Direct - matrix of directions
 * @param [out] one_image_features - vector of picture descriptor and it's class number
 * @param [in] Abs - matrix of gradient absolut values
 */
void to_features(Matrix<float> &Direct, Matrix<float> &Abs, vector<float> &one_image_features){

    int h = Direct.n_rows, w = Direct.n_cols;
    float d_pi = 2 * PI/pi_seg_num, d_w = w / hor_seg_num, d_h = h / vert_seg_num;
        int first_hor_pix, first_vert_pix, last_hor_pix = -1, last_vert_pix = -1;
        for (int ind_h = 0; ind_h < vert_seg_num; ++ind_h) {
            first_vert_pix = last_vert_pix + 1;
            if ( ind_h == vert_seg_num - 1)
                last_vert_pix = h - 1;
            else
                last_vert_pix = round((ind_h + 1) * d_h);
            last_hor_pix = -1;
            for (int ind_w = 0; ind_w < hor_seg_num; ++ind_w){
                first_hor_pix = last_hor_pix + 1;
                if ( ind_w == hor_seg_num - 1)
                    last_hor_pix = w - 1;
                else
                    last_hor_pix = round((ind_w + 1) * d_w);
                vector<float> histogram;
                histogram.resize(pi_seg_num);
                for (int i = 0; i < pi_seg_num; ++i)
                    histogram.at(i) = 0;
                Matrix<float> Sub_direct = Direct.submatrix(first_vert_pix, first_hor_pix, last_vert_pix - first_vert_pix + 1, last_hor_pix - first_hor_pix + 1);
                Matrix<float> Sub_abs = Abs.submatrix(first_vert_pix, first_hor_pix,  last_vert_pix - first_vert_pix + 1, last_hor_pix - first_hor_pix + 1);
                uint sub_h = Sub_abs.n_rows, sub_w = Sub_abs.n_cols;
                for (uint i = 0; i < sub_h; ++i)
                    for (uint j = 0; j < sub_w; ++ j){
                        float carry = -PI, num = 0;
                        while ((carry + (num + 1) * d_pi < Sub_direct(i, j)) && (num < pi_seg_num - 1))
                            ++num;
                        histogram.at(num) += Sub_abs(i, j);
                    }
                float norm = 0;
                for (int i = 0; i < pi_seg_num; ++i)
                    norm += histogram.at(i) * histogram[i];
                norm = sqrt(norm);
                if (abs(norm) > 0.00000001)
                    for (int i = 0; i < pi_seg_num; ++i)
                        histogram.at(i) /= norm;
               one_image_features.insert(one_image_features.end(), histogram.begin(), histogram.end());
            }
        }
}

/** Exatract features from dataset.
 * @param [in] data_set - data set with images and their class number
 * @param [out] features - vector of picture descriptor and it's class number
 * @param [in] sse - enables or disables sse usage
 */

void ExtractFeatures(const TDataSet& data_set, TFeatures* features, bool sse_on) {
    Timer t;
    t.start();
    for (size_t image_idx = 0; image_idx < data_set.size(); ++image_idx) {
        vector<float> one_image_features;
        BMP *im = data_set[image_idx].first;
        uint h = im->TellHeight() - 2, w = im->TellWidth() - 2;
        Matrix <float>  Direct(h, w), Abs(h, w);
        if (sse_on){
            get_dir_abs_SSE(im, Direct, Abs);
            to_features(Direct, Abs, one_image_features);
        }
        else{
            get_dir_abs(im, Direct, Abs);
            to_features(Direct, Abs, one_image_features);
        }
        features->push_back(make_pair(one_image_features,  data_set[image_idx].second));
    }
    if (sse_on)
        t.check("SSE ON");
    else
        t.check("SSE OFF");
}

/**Clear dataset structure
*/
void ClearDataset(TDataSet* data_set) {
        // Delete all images from dataset
    for (size_t image_idx = 0; image_idx < data_set->size(); ++image_idx)
        delete (*data_set)[image_idx].first;
        // Clear dataset
    data_set->clear();
}

/** Train SVM classifier using data from 'data_file' and save trained model to 'model_file'
*/
void TrainClassifier(const string& data_file, const string& model_file, bool sse_on) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // Model which would be trained
    TModel model;
        // Parameters of classifier
    TClassifierParams params;

        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features, sse_on);

        // PLACE YOUR CODE HERE
        // You can change parameters of classifier here
    params.C = 0.01;
    TClassifier classifier(params);
        // Train classifier
    classifier.Train(features, &model);
        // Save model to file
    model.Save(model_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

/**Predict data from 'data_file' using model from 'model_file' and save predictions to 'prediction_file'
*/
void PredictData(const string& data_file,
                 const string& model_file,
                 const string& prediction_file, bool sse_on) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // List of image labels
    TLabels labels;

        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features, sse_on);

        // Classifier
    TClassifier classifier = TClassifier(TClassifierParams());
        // Trained model
    TModel model;
        // Load model from file
    model.Load(model_file);
        // Predict images by its features using 'model' and store predictions
        // to 'labels'
    classifier.Predict(features, model, &labels);

        // Save predictions
    SavePredictions(file_list, labels, prediction_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

/**
@function Get4Pixels16Bit
reads four consecutive pixels of the specified row started from given column and writes they to the
two registers out_BG and out_RA. Uses 16 bit per channel
@param in_img is a input image
@param in_row_idx is an index of a row to read pixels
@param in_col_idx ia an index of a column with a first pixel
@param out_BG is a pointer to a 128bit register to store blue and green channels for the pixels four
consecutive pixels in format BBBB GGGG. Order of pixels is [0, 1, 2, 3]
@param out_RA is a pointer to a 128bit register to store red and alpha channels for the pixels four
consecutive pixels in format RRRR AAAA. Order of pixels is [0, 1, 2, 3]
*/
int oldmain(int argc, char** argv) {
    // Command line options parser
    ArgvParser cmd;
        // Description of program
    cmd.setIntroductoryDescription("Machine graphics course, task 2. CMC MSU, 2014.");
        // Add help option
    cmd.setHelpOption("h", "help", "Print this help message");
        // Add other options
    cmd.defineOption("data_set", "File with dataset",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("model", "Path to file to save or load model",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("predicted_labels", "Path to file to save prediction results",
        ArgvParser::OptionRequiresValue);
    cmd.defineOption("train", "Train classifier");
    cmd.defineOption("predict", "Predict dataset");
    cmd.defineOption("sse", "Use SSE");

        // Add options aliases
    cmd.defineOptionAlternative("data_set", "d");
    cmd.defineOptionAlternative("model", "m");
    cmd.defineOptionAlternative("predicted_labels", "l");
    cmd.defineOptionAlternative("train", "t");
    cmd.defineOptionAlternative("predict", "p");
    cmd.defineOptionAlternative("sse", "s");
        // Parse options
    int result = cmd.parse(argc, argv);

        // Check for errors or help option
    if (result) {
        cout << cmd.parseErrorDescription(result) << endl;
        return result;
    }

        // Get values
    string data_file = cmd.optionValue("data_set");
    string model_file = cmd.optionValue("model");
    bool train = cmd.foundOption("train");
    bool predict = cmd.foundOption("predict");
    bool sse_on = cmd.foundOption("sse");
        // If we need to train classifier
    if (train)
        TrainClassifier(data_file, model_file, sse_on);
        // If we need to predict data
    if (predict) {
            // You must declare file to save images
        if (!cmd.foundOption("predicted_labels")) {
            cerr << "Error! Option --predicted_labels not found!" << endl;
            return 1;
        }
            // File to save predictions
        string prediction_file = cmd.optionValue("predicted_labels");
            // Predict data
        PredictData(data_file, model_file, prediction_file, sse_on);
    }
    return 0;
}
