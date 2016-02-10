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


typedef unsigned char uchar;
#define PI 3.14159265
#define pi_seg_num 8
#define vert_seg_num 4
#define hor_seg_num 8
#define L 0.35
/// Weight of red channel multiplied by 256. It's stored as 8 equal 16 bit integer values
static const __m128i CONST_RED_16_INT = _mm_set1_epi16(static_cast<short>(0.2125f * 256));
/// Weight of green channel multiplied by 256. It's stored as 8 equal 16 bit integer values
static const __m128i CONST_GREEN_16_INT = _mm_set1_epi16(static_cast<short>(0.7154f * 256));
/// Weight of blue channel multiplied by 256. It's stored as 8 equal 16 bit integer values
static const __m128i CONST_BLUE_16_INT = _mm_set1_epi16(static_cast<short>(0.0721f * 256));

/**
@class MyMat
Simple matrix realization. Allows implicit allocating and deallocating memory for the matrix.
Gives C-style interface for memory access. Matrix is stored row-by-row.
Each element can contain few numbers.
For example, MyMat<uchar> img(N, M, 4) can store RGBA image with N rows and M columns.
<ul>
<li>To access the <i>i<sup>th</sup></i> row of the matrix use
@code (T*)((char*)data + i * step) @endcode
<li>To access the <i>j<sup>th</sup></i> element in the <i>i<sup>th</sup></i> row use
@code (T*)((char*)data + i * step) + j * channels @endcode
<li>To access the <i>k<sup>th</sup></i> channel of the <i>j<sup>th</sup></i> element
in the <i>i<sup>th</sup></i> row use
@code (T*)((char*)data + i * step) + j * channels @endcode
</ul>
*/
template<class T>
class MyMat
{
//T* olddata = nullptr;
public:
  /// stores pointer to the data
  T *data;
  /// stores number of rows in the matrix
  size_t rows;

  /// stores number of columns in the matrix
  size_t cols;

  /// stores number of bytes between two consecutive rows of the matrix. step >= cols * channels * sizeof(T)
  size_t step;

  /// stores number of channels per element
  size_t channels;

  /// default constructor. Constructs empty matrix
  MyMat()
    : rows(0), cols(0), channels(0), step(0), data(nullptr)
  {
  }

  /// constructor.
  /// Constructs matrix with specified number of rows, columns and channels per element
  /// @param in_rows is a number of rows in the matrix
  /// @param in_cols is a number of columns in the matrix
  /// @param in_channels is a number of channels per element
  MyMat(size_t in_rows, size_t in_cols, size_t in_channels)
  {
    data = nullptr;
    Init(in_rows, in_cols, in_channels);
  }

  /// destructor.
  /// Deallocates used memory
  ~MyMat()
  {
    clear();
  }

  /// Constructs matrix with specified number of rows, columns and channels per element
  /// @param in_rows is a number of rows in the matrix
  /// @param in_cols is a number of columns in the matrix
  /// @param in_channels is a number of channels per element
  void Init(size_t in_rows, size_t in_cols, size_t in_channels)
  {
    clear();
    rows = in_rows;
    cols = in_cols;
    channels = in_channels;
    size_t elems_in_row = cols * channels;
    size_t elems_in_mat = elems_in_row * in_rows;
    data = new T[elems_in_mat];
    step = elems_in_row * sizeof(T);
  }

private:
  /// Deallocates used memory
  void clear()
  {
    if (data != nullptr)
    {
      delete [] data;
      data = nullptr;
    }
  }

  /// Copy constructor. Copying is not allowed.
  MyMat(const MyMat &in_obj)
  {
  }
};


typedef vector<pair<BMP*, int> > TDataSet;
typedef vector<pair<string, int> > TFileList;
typedef vector<pair<vector<float>, int> > TFeatures;
typedef Matrix<std::tuple<uint, uint, uint>> Image;
int oldmain(int argc, char** argv);
void PredictData(const string& data_file,
                 const string& model_file,
                 const string& prediction_file);
void TrainClassifier(const string& data_file, const string& model_file);
void ClearDataset(TDataSet* data_set);
void ExtractFeatures(const TDataSet& data_set, TFeatures* features);
void SavePredictions(const TFileList& file_list,
                     const TLabels& labels,
                     const string& prediction_file);
void LoadImages(const TFileList& file_list, TDataSet* data_set);
void LoadFileList(const string& data_file, TFileList* file_list);
void toGrayScaleSSE_16BIT(BMP &in_input, MyMat<uchar> &out_mat);
inline void Get4Pixels16Bit(BMP &in_img, int in_row_idx, int in_col_idx,
                            __m128i *out_BG, __m128i *out_RA);
void get_dir_abs_SSE(BMP *im, Matrix<float> &Direct, Matrix<float> &Abs);
void get_dir_abs(BMP *im, Matrix<float> &Direct, Matrix<float> &Abs);
void to_features(Matrix<float> &Direct, Matrix<float> &Abs, vector<float> &one_image_features);
int oldmain(int argc, char** argv);
