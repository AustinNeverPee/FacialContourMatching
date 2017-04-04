#include <iostream>
#include <fstream>
#include <cv.h>
#include <highgui.h>

using namespace std;
using namespace cv;

#define M_PI 3.14159265358979323846

// 使用的点的个数
const static int point_num = 16;
// 定义图像的大小
const static int img_width = 600, img_height = 600;
// 定义轮廓点坐标
static int **points = new int*[45];
// 定义坐标缩放尺度
const static double scale[10] = {0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2 };

// 定义轮廓结构体
struct Outline
{
	double value;
	int index;
};
Outline outlines_simi[45];
Outline outlines_diff[45];
Outline scores[45];

// 交换两个结构体
void Swap(int flag, int index_A, int index_B)
{
	Outline tmp;
	if (flag == 1)
	{
		tmp.value = outlines_simi[index_A].value;
		tmp.index = outlines_simi[index_A].index;
		outlines_simi[index_A].value = outlines_simi[index_B].value;
		outlines_simi[index_A].index = outlines_simi[index_B].index;
		outlines_simi[index_B].value = tmp.value;
		outlines_simi[index_B].index = tmp.index;
	}
	else if (flag == 2)
	{
		tmp.value = outlines_diff[index_A].value;
		tmp.index = outlines_diff[index_A].index;
		outlines_diff[index_A].value = outlines_diff[index_B].value;
		outlines_diff[index_A].index = outlines_diff[index_B].index;
		outlines_diff[index_B].value = tmp.value;
		outlines_diff[index_B].index = tmp.index;
	}
	else if (flag == 3)
	{
		tmp.value = scores[index_A].value;
		tmp.index = scores[index_A].index;
		scores[index_A].value = scores[index_B].value;
		scores[index_A].index = scores[index_B].index;
		scores[index_B].value = tmp.value;
		scores[index_B].index = tmp.index;
	}
}

/***************** Procrustes distance匹配 *****************/

//  输出矩阵内容
void showcvmat(CvMat* mat, const char* prompt) {
	/*           判断是否是高纬度的矩阵           */
	if (mat->rows == -1 || mat->cols == -1) {
		printf("矩阵超过2维，无法输出...\n");
	}

	/*               输出提示命令                */
	if (prompt == NULL) {
		printf("未知名内容\n");
	}
	else {
		printf("%s\n", prompt);
	}

	/*             输出矩阵的函数和列数           */
	printf("行数:%d\t列数:%d\n", mat->rows, mat->cols);

	/*            遍历输出矩阵结果                */
	for (int i = 0; i < mat->rows; ++i) {
		for (int j = 0; j < mat->cols; ++j) {
			float text = CV_MAT_ELEM(*mat, float, i, j);
			if (j == mat->cols - 1) {
				printf("%f\n", text);
			}
			else {
				printf("%f\t", text);
			}
		}
	}
}

/* 定义复数向量的基本运算，理论上是支持复数矩阵的 */
void complexmuti(CvMat* Wr, CvMat* Wv, CvMat* Zr, CvMat* Zv, CvMat* WZr, CvMat* WZv) {
	//printf("start calculating complex...\n");
	assert(Wr != NULL && Wv != NULL && Zr != NULL && Zv != NULL);
	assert(Wr->rows == Wv->rows && Wr->cols == Wv->cols);
	assert(Zr->rows == Zv->rows && Zr->cols == Zv->cols);
	assert(Wr->cols == Zr->rows && Wv->cols == Zv->rows);

	/* 计算过程中用到的子矩阵 */
	CvMat* WrZr = cvCreateMat(Wr->rows, Zr->cols, CV_32FC1);
	// 如果用cvMul函数，会触发通道异常
	//printf("cvmatmul...\n");
	cvMatMul(Wr, Zr, WrZr);
	//printf("ending cvmatmul ....\n");
	CvMat* WvZv = cvCreateMat(Wv->rows, Zv->cols, CV_32FC1);
	cvMatMul(Wv, Zv, WvZv);
	CvMat* WrZv = cvCreateMat(Wr->rows, Zv->cols, CV_32FC1);
	cvMatMul(Wr, Zv, WrZv);
	CvMat* WvZr = cvCreateMat(Wv->rows, Zr->cols, CV_32FC1);
	cvMatMul(Wv, Zr, WvZr);

	/* 利用子矩阵合成复数矩阵 */
	cvSub(WrZr, WvZv, WZr);
	cvAdd(WrZv, WvZr, WZv);

	//printf("end calculating complex...\n");
}

// 复数向量的模计算理论有问题，这里不要模的运算。
// A x A的共轭矩阵，然后，对每一列求sum
float complexmode(CvMat* R, CvMat* V) {
	assert(R != NULL && V != NULL);
	float partr = 0.0, partv = 0.0;
	for (int i = 0; i < R->rows; ++i) {
		partr += CV_MAT_ELEM(*R, float, i, 0);
		partv += CV_MAT_ELEM(*V, float, i, 0);
	}

	return sqrt(pow(partr, 2) + pow(partv, 2));
}

// 对矩阵单独的列进行正规化
// 移除缩放变换
void avragevector(CvMat* mat, int col) {
	float* head = (float*)CV_MAT_ELEM_PTR(*mat, 0, 0);
	float min = 10000000.0, max = 0.0;
	for (int i = 0; i < mat->rows; ++i) {
		float tp = CV_MAT_ELEM(*mat, float, i, col);
		if (tp > max) max = tp;
		if (tp < min) min = tp;
	}

	for (int i = 0; i < mat->rows; ++i) {
		float* ptr = (float*)CV_MAT_ELEM_PTR(*mat, i, col);
		*ptr = (*ptr - min) / (max - min);
	}
}

// 计算df因子必须分离出来, wr k * 1,  zr 1 * k
float CalDF(int* points_A, int* points_B)
{
	// 计算Procrustes distance使用变量
	float* datar2 = new float[point_num];
	float* datav2 = new float[point_num];
	float* datar1 = new float[point_num];
	float* datav1 = new float[point_num];
	for (int j = 0; j < point_num * 2; j += 2)
	{
		datar1[j / 2] = points_A[j];
		datav1[j / 2] = points_A[j + 1];
		datar2[j / 2] = points_B[j];
		datav2[j / 2] = points_B[j + 1];
	}

	CvMat matwr = cvMat(point_num, 1, CV_32FC1, datar1);
	avragevector(&matwr, 0);
	CvMat matwv = cvMat(point_num, 1, CV_32FC1, datav1);
	avragevector(&matwv, 0);
	CvMat matzr = cvMat(point_num, 1, CV_32FC1, datar2);
	avragevector(&matzr, 0);
	CvMat matzv = cvMat(point_num, 1, CV_32FC1, datav2);
	avragevector(&matzv, 0);

	CvMat *wr = &matwr, *wv = &matwv, *zr = &matzr, *zv = &matzv;

	// w的共轭转置
	CvMat* wr_T = cvCreateMat(wr->cols, wr->rows, CV_32FC1);
	cvTranspose(wr, wr_T);
	CvMat* wv_T = cvCreateMat(wv->cols, wv->rows, CV_32FC1);
	cvTranspose(wv, wv_T);
	float* head = (float*)CV_MAT_ELEM_PTR(*wv_T, 0, 0);
	for (int i = 0; i < wv_T->rows * wv_T->cols; ++i) {
		*head = -(*head);
		++head;
	}

	// z的共轭转置
	CvMat* zr_T = cvCreateMat(zr->cols, zr->rows, CV_32FC1);
	cvTranspose(zr, zr_T);
	CvMat* zv_T = cvCreateMat(zv->cols, zv->rows, CV_32FC1);
	cvTranspose(zv, zv_T);
	head = (float*)CV_MAT_ELEM_PTR(*zv_T, 0, 0);
	for (int i = 0; i < zv_T->rows * zv_T->cols; ++i) {
		*head = -(*head);
		++head;
	}

	//    计算分子的值
	CvMat* resr = cvCreateMat(1, 1, CV_32FC1);
	CvMat* resv = cvCreateMat(1, 1, CV_32FC1);
	complexmuti(wr_T, wv_T, zr, zv, resr, resv);
	float realpart = CV_MAT_ELEM(*resr, float, 0, 0);
	float virpart = CV_MAT_ELEM(*resv, float, 0, 0);
	// 这里平方之后出现数据越界，变为NAN
	float numerator = pow(realpart, 2) + pow(virpart, 2);
	//printf("%f + i %f; numertor:%f\n", realpart, virpart, numerator);

	// 计算分母的值
	complexmuti(wr_T, wv_T, wr, wv, resr, resv);
	float leftrpart = CV_MAT_ELEM(*resr, float, 0, 0);
	float leftvpart = CV_MAT_ELEM(*resv, float, 0, 0);
	complexmuti(zr_T, zv_T, zr, zv, resr, resv);
	float rightrpart = CV_MAT_ELEM(*resr, float, 0, 0);
	float rightvpart = CV_MAT_ELEM(*resv, float, 0, 0);

	float denominator = leftrpart * rightrpart;
	/*printf("lr:%f\t, lv:%f\t, rr:%f\t, rv:%f\n", leftrpart, leftvpart, rightrpart, rightvpart);
	printf("%f\t /	%f\n", numerator, denominator);*/

	// 垃圾回收
	delete[] datar1;
	delete[] datav1;
	delete[] datar2;
	delete[] datav2;

	return 1.0 - numerator / denominator;
}


// 对点集做归一化处理
float CalDFFactor(int *_points_A, int *_points_B) {
	int points_A[point_num * 2], points_B[point_num * 2];
	for (int i = 0; i < point_num * 2; i++)
	{
		points_A[i] = _points_A[i];
		points_B[i] = _points_B[i];
	}

	// 移除平移变换
	int x_A = points_A[12];
	int y_A = points_A[1];
	int x_B = points_B[12];
	int y_B = points_B[1];
	for (int i = 0; i < point_num; i++)
	{
		// j轮廓的中心
		points_A[i * 2] -= x_A;
		points_A[i * 2 + 1] -= y_A;
		points_B[i * 2] -= x_B;
		points_B[i * 2 + 1] -= y_B;
	}

	// 移除旋转变换
	// 计算不同旋转角度下的差别，选取最小的
	float diff = 1;
	int points_tmp[point_num * 2];
	for (int j = 1; j < 25; j++)
	{
		int angle = 15 * j;
		double rotate = M_PI * angle / 180;
		for (int i = 0; i < point_num; i++)
		{
			points_tmp[i * 2] = points_B[i * 2] * cos(rotate) - points_B[i * 2 + 1] * sin(rotate);
			points_tmp[i * 2 + 1] = points_B[i * 2 + 1] * cos(rotate) + points_B[i * 2] * sin(rotate);
		}
		float diff_tmp = CalDF(points_A, points_tmp);
		diff = diff_tmp < diff ? diff_tmp : diff;
	}

	return diff;
}
/***************** Procrustes distance匹配 *****************/


/***************** Hu矩匹配 *****************/
/**
* 不使用opencv中的Hu矩函数
**/
double* CalHu(int *points, double** img)
{
	// 更新图片
	for (int i = 0; i < img_height; i++)
	{
		for (int j = 0; j < img_width; j++)
		{
			img[i][j] = 0;
		}
	}
	for (int i = 0; i < point_num; i++)
	{
		img[points[i * 2 + 1]][points[i * 2]] = 1;
	}

	// 计算Hu矩
	// 计算普通矩
	double m00 = 0, m01 = 0, m10 = 0;
	for (int i = 0; i < img_height; i++)
	{
		for (int j = 0; j < img_width; j++)
		{
			m00 += img[i][j];
			m01 += i * img[i][j];
			m10 += j * img[i][j];
		}
	}

	// 计算区域重心
	double x0 = m10 / m00;
	double y0 = m01 / m00;

	// 计算中心矩
	double u00 = m00, u02 = 0, u03 = 0, u11 = 0, u12 = 0, u20 = 0, u21 = 0, u30 = 0;
	for (int i = 0; i < img_height; i++)
	{
		for (int j = 0; j < img_width; j++)
		{
			u02 += (i - y0) * (i - y0) * img[i][j];
			u03 += (i - y0) * (i - y0) * (i - y0) * img[i][j];
			u11 += (j - x0) * (i - y0) * img[i][j];
			u12 += (j - x0) * (i - y0) * (i - y0) * img[i][j];
			u20 += (j - x0) * (j - x0) * img[i][j];
			u21 += (j - x0) * (j - x0) * (i - y0) * img[i][j];
			u30 += (j - x0) * (j - x0) * (j - x0) * img[i][j];
		}
	}

	// 计算归一化中心矩
	double y02 = u02 / pow(u00, 2);
	double y03 = u03 / pow(u00, 2.5);
	double y11 = u11 / pow(u00, 2);
	double y12 = u12 / pow(u00, 2.5);
	double y20 = u20 / pow(u00, 2);
	double y21 = u21 / pow(u00, 2.5);
	double y30 = u30 / pow(u00, 2.5);

	// 计算中心变量
	double t1 = (y20 - y02);
	double t2 = (y30 - 3 * y12);
	double t3 = (3 * y21 - y03);
	double t4 = (y30 + y12);
	double t5 = (y21 + y03);

	// 计算不变矩
	double *I = new double[7];
	I[0] = y20 + y02;
	I[1] = t1 * t1 + 4 * y11 * y11;
	I[2] = t2 * t2 + t3 * t3;
	I[3] = t4 * t4 + t5 * t5;
	I[4] = t2 * t4 * (t4 * t4 - 3 * t5 * t5) + t3 * t5 * (3 * t4 * t4 - t5 * t5);
	I[5] = t1 * (t4 * t4 - t5 * t5) + 4 * y11 * t4 * t5;
	I[6] = t3 * t4 * (t4 * t4 - 3 * t5 * t5) - t2 * t5 * (3 * t4 * t4 - t5 * t5);

	return I;
}

double CalContSimi(int *points_A, int *points_B) {
	// 根据ASM得出的点，创建“二维”图像（img[y][x]，与points横纵坐标顺序相反）
	double** img = new double*[img_height];
	for (int i = 0; i < img_height; i++)
	{
		img[i] = new double[img_width];
	}

	double *I_A = CalHu(points_A, img);
	double *I_B = CalHu(points_B, img);

	// 计算两个轮廓相似度
	double similarity, tmp1 = 0, tmp2 = 0;
	for (int i = 0; i < 7; i++)
	{
		tmp1 += fabs(I_A[i] - I_B[i]);
		tmp2 += fabs(I_A[i] + I_B[i]);
	}

	similarity = 1 - tmp1 / tmp2;

	// 垃圾回收
	for (int i = 0; i < img_height; i++)
	{
		delete[] img[i];
	}
	delete[] img;

	return similarity;
}
/***************** Hu矩匹配 *****************/


/***************** 综合两种匹配得分排序 *****************/
int MatchRank(int *points_A)
{
	for (int j = 0; j < 45; j++)
	{
		// 计算Hu矩相似度 & Procrustes distance差别
		double simi = 0, diff = 1;
		// 计算不同尺度下的匹配度，挑选最好的匹配
		for (int m = 0; m < 10; m++)
		{
			int point_tmp[point_num * 2];
			bool out_of_range = false;
			for (int n = 0; n < point_num; n++)
			{
				point_tmp[n * 2] = points[j][n * 2] * scale[m];
				point_tmp[n * 2 + 1] = points[j][n * 2 + 1] * scale[m];

				if (point_tmp[n * 2] >= 600 || point_tmp[n * 2 + 1] >= 600)
				{
					out_of_range = true;
					break;
				}
			}
			if (out_of_range)
			{
				break;
			}
			else
			{
				double simi_tmp = CalContSimi(points_A, point_tmp);
				double diff_tmp = CalDFFactor(points_A, point_tmp);
				simi = simi_tmp > simi ? simi_tmp : simi;
				diff = diff_tmp < diff ? diff_tmp : diff;
			}
		}
		outlines_simi[j].value = simi;
		outlines_simi[j].index = j;
		// 计算
		outlines_diff[j].value = diff;
		outlines_diff[j].index = j;

		// 排序得分清零
		scores[j].value = 0;
		scores[j].index = j;
	}

	// 按照Hu矩排序（降序）
	for (int m = 0; m < 44; m++)
	{
		for (int n = 0; n < 44 - m; n++)
		{
			if (outlines_simi[n].value < outlines_simi[n + 1].value)
			{
				Swap(1, n, n + 1);
			}
		}
	}

	// 按照Procrustes distance排序（升序）
	for (int m = 0; m < 44; m++)
	{
		for (int n = 0; n < 44 - m; n++)
		{
			if (outlines_diff[n].value > outlines_diff[n + 1].value)
			{
				Swap(2, n, n + 1);
			}
		}
	}

	// 通过两种排名计算轮廓匹配得分
	// 排名越靠前，分量越足
	for (int m = 0; m < 45; m++)
	{
		scores[outlines_simi[m].index].value += (m + 1) * (m + 1);
		scores[outlines_diff[m].index].value += (m + 1) * (m + 1);
	}

	// 对得分排序，选出最匹配轮廓
	for (int m = 0; m < 44; m++)
	{
		for (int n = 0; n < 44 - m; n++)
		{
			if (scores[n].value > scores[n + 1].value)
			{
				Swap(3, n, n + 1);
			}
		}
	}

	/*cout << "Hu矩排序（降序）：" << endl;
	for (int m = 0; m < 45; m++)
	{
		cout << m << "." << outlines_simi[m].index << "<<" << outlines_simi[m].value << "  ";
	}
	cout << endl
		<< "Procrustes distance排序（升序）：" << endl;
	for (int m = 0; m < 45; m++)
	{
		cout << m << "." << outlines_diff[m].index << "<<" << outlines_diff[m].value << "  ";
	}
	cout << endl
		<< "综合两种排序得分（升序）：" << endl;
	for (int m = 0; m < 45; m++)
	{
		cout << m + 1 << "." << scores[m].index << "<<" << scores[m].value << "  ";
	}
	cout << endl;*/

	return scores[0].index;
}
/***************** 综合两种匹配得分排序 *****************/

//// C1与C1匹配
//int main()
//{
//	// 从文本中读入坐标进行计算
//	ifstream infile;
//	infile.open("E:/大四下/微漫/形状匹配/素材/测试图片/facePoint.txt");
//	for (int i = 0; i < 45; i++)
//	{
//		points[i] = new int[32];
//		for (int j = 0; j < 32; j++)
//		{
//			infile >> points[i][j];
//		}
//	}
//	infile.close();
//
//
//	ofstream outfile;
//	outfile.open("E:/大四下/微漫/形状匹配/Hu矩_Procrustes distance匹配/输出结果/1_1/result.txt");
//	// 对所有素材进行轮廓匹配，观察期匹配结果
//	int count = 0;
//	for (int i = 0; i < 45; i++)
//	{
//		int best_index = MatchRank(points[i]);
//		if (i < 10)
//		{
//			cout << "轮廓" << i << "（C1）        匹配到        轮廓" << best_index << "（C1）" << endl;
//		}
//		else
//		{
//			cout << "轮廓" << i << "（C1）       匹配到        轮廓" << best_index << "（C1）" << endl;
//		}
//		outfile << i << "	<<	" << best_index << endl;
//
//
//		// 画出最匹配的两个轮廓
//		// 将两个轮廓中心平移到一起
//		// i轮廓的中心
//		int x_i = points[i][12];
//		int y_i = points[i][1];
//		// j轮廓的中心
//		int x_j = points[best_index][12];
//		int y_j = points[best_index][1];
//		// 移动距离
//		int x_move = x_j - x_i;
//		int y_move = y_j - y_i;
//		// 平移
//		int points_j[32];
//		for (int j = 0; j < 16; j++)
//		{
//			points_j[j * 2] = points[best_index][j * 2] - x_move;
//			points_j[j * 2 + 1] = points[best_index][j * 2 + 1] - y_move;
//		}
//
//		Mat picture(img_width, img_height, CV_8UC3, Scalar(255, 255, 255));
//		for (int j = 0; j < 16; j++)
//		{
//			// 原图为黑色
//			circle(picture, Point(points[i][j * 2], points[i][j * 2 + 1]), 2, CV_RGB(0, 0, 0), 2, 8, 0);
//			// 匹配图为红色
//			circle(picture, Point(points_j[j * 2], points_j[j * 2 + 1]), 2, CV_RGB(255, 0, 0), 2, 8, 0);
//		}
//
//		// 保存轮廓对比图像
//		string pic_name = "E:/大四下/微漫/形状匹配/Hu矩_Procrustes distance匹配/输出结果/1_1/";
//		stringstream ss;
//		string str;
//		ss << i;
//		ss >> str;
//		pic_name.append(str + ".jpg");
//		imwrite(pic_name, picture);
//
//		// imshow("画板" + i, picture);
//	}
//
//	outfile.close();
//
//	waitKey(0);
//	system("pause");
//
//	return 0;
//}


// C2与C1匹配
int main()
{
	// 从文本中读入坐标进行计算
	ifstream infile;
	infile.open("E:/大四下/微漫/形状匹配/素材/测试图片/facePoint.txt");
	for (int i = 0; i < 45; i++)
	{
		points[i] = new int[32];
		for (int j = 0; j < 32; j++)
		{
			infile >> points[i][j];
		}
	}
	infile.close();
	infile.open("E:/大四下/微漫/形状匹配/素材/测试图片/facePointNew.txt");
	int **points_new = new int*[45];
	for (int i = 0; i < 45; i++)
	{
		points_new[i] = new int[32];
		for (int j = 0; j < 32; j++)
		{
			infile >> points_new[i][j];
		}
	}
	infile.close();


	ofstream outfile;
	outfile.open("E:/大四下/微漫/形状匹配/Hu矩_Procrustes distance匹配/输出结果/1_2/result.txt");
	// 对所有素材进行轮廓匹配，观察期匹配结果
	int count = 0;
	for (int i = 0; i < 45; i++)
	{
		int best_index = MatchRank(points_new[i]);
		if (i < 10)
		{
			cout << "轮廓" << i << "（C2）        匹配到        轮廓" << best_index << "（C1）" << endl;
		}
		else
		{
			cout << "轮廓" << i << "（C2）       匹配到        轮廓" << best_index << "（C1）" << endl;
		}
		outfile << i << "	<<	" << best_index << endl;


		// 画出最匹配的两个轮廓
		// 将两个轮廓中心平移到一起
		// i轮廓的中心
		int x_i = points_new[i][12];
		int y_i = points_new[i][1];
		// j轮廓的中心
		int x_j = points[best_index][12];
		int y_j = points[best_index][1];
		// 移动距离
		int x_move = x_j - x_i;
		int y_move = y_j - y_i;
		// 平移
		int points_j[32];
		for (int j = 0; j < 16; j++)
		{
			points_j[j * 2] = points[best_index][j * 2] - x_move;
			points_j[j * 2 + 1] = points[best_index][j * 2 + 1] - y_move;
		}

		Mat picture(img_width, img_height, CV_8UC3, Scalar(255, 255, 255));
		for (int j = 0; j < 16; j++)
		{
			// 原图为黑色
			circle(picture, Point(points_new[i][j * 2], points_new[i][j * 2 + 1]), 2, CV_RGB(0, 0, 0), 2, 8, 0);
			// 匹配图为红色
			circle(picture, Point(points_j[j * 2], points_j[j * 2 + 1]), 2, CV_RGB(255, 0, 0), 2, 8, 0);
		}

		// 保存轮廓对比图像
		string pic_name = "E:/大四下/微漫/形状匹配/Hu矩_Procrustes distance匹配/输出结果/1_2/";
		stringstream ss;
		string str;
		ss << i;
		ss >> str;
		pic_name.append(str + ".jpg");

		// 在图片中显示标号
		putText(picture, str, Point(400, 150), CV_FONT_HERSHEY_DUPLEX, 5.0f, CV_RGB(0, 0, 0));
		imwrite(pic_name, picture);

		// imshow("画板" + i, picture);
		if (i == best_index)
		{
			count++;
		}
	}

	double rate = (double)count / 45;
	cout << endl << "匹配正确率为" << rate * 100 << "%" << endl << endl;

	outfile.close();

	waitKey(0);
	system("pause");

	return 0;
}