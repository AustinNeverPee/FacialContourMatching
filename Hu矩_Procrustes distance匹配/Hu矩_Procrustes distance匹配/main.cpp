#include <iostream>
#include <fstream>
#include <cv.h>
#include <highgui.h>

using namespace std;
using namespace cv;

#define M_PI 3.14159265358979323846

// ʹ�õĵ�ĸ���
const static int point_num = 16;
// ����ͼ��Ĵ�С
const static int img_width = 600, img_height = 600;
// ��������������
static int **points = new int*[45];
// �����������ų߶�
const static double scale[10] = {0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2 };

// ���������ṹ��
struct Outline
{
	double value;
	int index;
};
Outline outlines_simi[45];
Outline outlines_diff[45];
Outline scores[45];

// ���������ṹ��
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

/***************** Procrustes distanceƥ�� *****************/

//  �����������
void showcvmat(CvMat* mat, const char* prompt) {
	/*           �ж��Ƿ��Ǹ�γ�ȵľ���           */
	if (mat->rows == -1 || mat->cols == -1) {
		printf("���󳬹�2ά���޷����...\n");
	}

	/*               �����ʾ����                */
	if (prompt == NULL) {
		printf("δ֪������\n");
	}
	else {
		printf("%s\n", prompt);
	}

	/*             �������ĺ���������           */
	printf("����:%d\t����:%d\n", mat->rows, mat->cols);

	/*            �������������                */
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

/* ���帴�������Ļ������㣬��������֧�ָ�������� */
void complexmuti(CvMat* Wr, CvMat* Wv, CvMat* Zr, CvMat* Zv, CvMat* WZr, CvMat* WZv) {
	//printf("start calculating complex...\n");
	assert(Wr != NULL && Wv != NULL && Zr != NULL && Zv != NULL);
	assert(Wr->rows == Wv->rows && Wr->cols == Wv->cols);
	assert(Zr->rows == Zv->rows && Zr->cols == Zv->cols);
	assert(Wr->cols == Zr->rows && Wv->cols == Zv->rows);

	/* ����������õ����Ӿ��� */
	CvMat* WrZr = cvCreateMat(Wr->rows, Zr->cols, CV_32FC1);
	// �����cvMul�������ᴥ��ͨ���쳣
	//printf("cvmatmul...\n");
	cvMatMul(Wr, Zr, WrZr);
	//printf("ending cvmatmul ....\n");
	CvMat* WvZv = cvCreateMat(Wv->rows, Zv->cols, CV_32FC1);
	cvMatMul(Wv, Zv, WvZv);
	CvMat* WrZv = cvCreateMat(Wr->rows, Zv->cols, CV_32FC1);
	cvMatMul(Wr, Zv, WrZv);
	CvMat* WvZr = cvCreateMat(Wv->rows, Zr->cols, CV_32FC1);
	cvMatMul(Wv, Zr, WvZr);

	/* �����Ӿ���ϳɸ������� */
	cvSub(WrZr, WvZv, WZr);
	cvAdd(WrZv, WvZr, WZv);

	//printf("end calculating complex...\n");
}

// ����������ģ�������������⣬���ﲻҪģ�����㡣
// A x A�Ĺ������Ȼ�󣬶�ÿһ����sum
float complexmode(CvMat* R, CvMat* V) {
	assert(R != NULL && V != NULL);
	float partr = 0.0, partv = 0.0;
	for (int i = 0; i < R->rows; ++i) {
		partr += CV_MAT_ELEM(*R, float, i, 0);
		partv += CV_MAT_ELEM(*V, float, i, 0);
	}

	return sqrt(pow(partr, 2) + pow(partv, 2));
}

// �Ծ��󵥶����н������滯
// �Ƴ����ű任
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

// ����df���ӱ���������, wr k * 1,  zr 1 * k
float CalDF(int* points_A, int* points_B)
{
	// ����Procrustes distanceʹ�ñ���
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

	// w�Ĺ���ת��
	CvMat* wr_T = cvCreateMat(wr->cols, wr->rows, CV_32FC1);
	cvTranspose(wr, wr_T);
	CvMat* wv_T = cvCreateMat(wv->cols, wv->rows, CV_32FC1);
	cvTranspose(wv, wv_T);
	float* head = (float*)CV_MAT_ELEM_PTR(*wv_T, 0, 0);
	for (int i = 0; i < wv_T->rows * wv_T->cols; ++i) {
		*head = -(*head);
		++head;
	}

	// z�Ĺ���ת��
	CvMat* zr_T = cvCreateMat(zr->cols, zr->rows, CV_32FC1);
	cvTranspose(zr, zr_T);
	CvMat* zv_T = cvCreateMat(zv->cols, zv->rows, CV_32FC1);
	cvTranspose(zv, zv_T);
	head = (float*)CV_MAT_ELEM_PTR(*zv_T, 0, 0);
	for (int i = 0; i < zv_T->rows * zv_T->cols; ++i) {
		*head = -(*head);
		++head;
	}

	//    ������ӵ�ֵ
	CvMat* resr = cvCreateMat(1, 1, CV_32FC1);
	CvMat* resv = cvCreateMat(1, 1, CV_32FC1);
	complexmuti(wr_T, wv_T, zr, zv, resr, resv);
	float realpart = CV_MAT_ELEM(*resr, float, 0, 0);
	float virpart = CV_MAT_ELEM(*resv, float, 0, 0);
	// ����ƽ��֮���������Խ�磬��ΪNAN
	float numerator = pow(realpart, 2) + pow(virpart, 2);
	//printf("%f + i %f; numertor:%f\n", realpart, virpart, numerator);

	// �����ĸ��ֵ
	complexmuti(wr_T, wv_T, wr, wv, resr, resv);
	float leftrpart = CV_MAT_ELEM(*resr, float, 0, 0);
	float leftvpart = CV_MAT_ELEM(*resv, float, 0, 0);
	complexmuti(zr_T, zv_T, zr, zv, resr, resv);
	float rightrpart = CV_MAT_ELEM(*resr, float, 0, 0);
	float rightvpart = CV_MAT_ELEM(*resv, float, 0, 0);

	float denominator = leftrpart * rightrpart;
	/*printf("lr:%f\t, lv:%f\t, rr:%f\t, rv:%f\n", leftrpart, leftvpart, rightrpart, rightvpart);
	printf("%f\t /	%f\n", numerator, denominator);*/

	// ��������
	delete[] datar1;
	delete[] datav1;
	delete[] datar2;
	delete[] datav2;

	return 1.0 - numerator / denominator;
}


// �Ե㼯����һ������
float CalDFFactor(int *_points_A, int *_points_B) {
	int points_A[point_num * 2], points_B[point_num * 2];
	for (int i = 0; i < point_num * 2; i++)
	{
		points_A[i] = _points_A[i];
		points_B[i] = _points_B[i];
	}

	// �Ƴ�ƽ�Ʊ任
	int x_A = points_A[12];
	int y_A = points_A[1];
	int x_B = points_B[12];
	int y_B = points_B[1];
	for (int i = 0; i < point_num; i++)
	{
		// j����������
		points_A[i * 2] -= x_A;
		points_A[i * 2 + 1] -= y_A;
		points_B[i * 2] -= x_B;
		points_B[i * 2 + 1] -= y_B;
	}

	// �Ƴ���ת�任
	// ���㲻ͬ��ת�Ƕ��µĲ��ѡȡ��С��
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
/***************** Procrustes distanceƥ�� *****************/


/***************** Hu��ƥ�� *****************/
/**
* ��ʹ��opencv�е�Hu�غ���
**/
double* CalHu(int *points, double** img)
{
	// ����ͼƬ
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

	// ����Hu��
	// ������ͨ��
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

	// ������������
	double x0 = m10 / m00;
	double y0 = m01 / m00;

	// �������ľ�
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

	// �����һ�����ľ�
	double y02 = u02 / pow(u00, 2);
	double y03 = u03 / pow(u00, 2.5);
	double y11 = u11 / pow(u00, 2);
	double y12 = u12 / pow(u00, 2.5);
	double y20 = u20 / pow(u00, 2);
	double y21 = u21 / pow(u00, 2.5);
	double y30 = u30 / pow(u00, 2.5);

	// �������ı���
	double t1 = (y20 - y02);
	double t2 = (y30 - 3 * y12);
	double t3 = (3 * y21 - y03);
	double t4 = (y30 + y12);
	double t5 = (y21 + y03);

	// ���㲻���
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
	// ����ASM�ó��ĵ㣬��������ά��ͼ��img[y][x]����points��������˳���෴��
	double** img = new double*[img_height];
	for (int i = 0; i < img_height; i++)
	{
		img[i] = new double[img_width];
	}

	double *I_A = CalHu(points_A, img);
	double *I_B = CalHu(points_B, img);

	// ���������������ƶ�
	double similarity, tmp1 = 0, tmp2 = 0;
	for (int i = 0; i < 7; i++)
	{
		tmp1 += fabs(I_A[i] - I_B[i]);
		tmp2 += fabs(I_A[i] + I_B[i]);
	}

	similarity = 1 - tmp1 / tmp2;

	// ��������
	for (int i = 0; i < img_height; i++)
	{
		delete[] img[i];
	}
	delete[] img;

	return similarity;
}
/***************** Hu��ƥ�� *****************/


/***************** �ۺ�����ƥ��÷����� *****************/
int MatchRank(int *points_A)
{
	for (int j = 0; j < 45; j++)
	{
		// ����Hu�����ƶ� & Procrustes distance���
		double simi = 0, diff = 1;
		// ���㲻ͬ�߶��µ�ƥ��ȣ���ѡ��õ�ƥ��
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
		// ����
		outlines_diff[j].value = diff;
		outlines_diff[j].index = j;

		// ����÷�����
		scores[j].value = 0;
		scores[j].index = j;
	}

	// ����Hu�����򣨽���
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

	// ����Procrustes distance��������
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

	// ͨ������������������ƥ��÷�
	// ����Խ��ǰ������Խ��
	for (int m = 0; m < 45; m++)
	{
		scores[outlines_simi[m].index].value += (m + 1) * (m + 1);
		scores[outlines_diff[m].index].value += (m + 1) * (m + 1);
	}

	// �Ե÷�����ѡ����ƥ������
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

	/*cout << "Hu�����򣨽��򣩣�" << endl;
	for (int m = 0; m < 45; m++)
	{
		cout << m << "." << outlines_simi[m].index << "<<" << outlines_simi[m].value << "  ";
	}
	cout << endl
		<< "Procrustes distance�������򣩣�" << endl;
	for (int m = 0; m < 45; m++)
	{
		cout << m << "." << outlines_diff[m].index << "<<" << outlines_diff[m].value << "  ";
	}
	cout << endl
		<< "�ۺ���������÷֣����򣩣�" << endl;
	for (int m = 0; m < 45; m++)
	{
		cout << m + 1 << "." << scores[m].index << "<<" << scores[m].value << "  ";
	}
	cout << endl;*/

	return scores[0].index;
}
/***************** �ۺ�����ƥ��÷����� *****************/

//// C1��C1ƥ��
//int main()
//{
//	// ���ı��ж���������м���
//	ifstream infile;
//	infile.open("E:/������/΢��/��״ƥ��/�ز�/����ͼƬ/facePoint.txt");
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
//	outfile.open("E:/������/΢��/��״ƥ��/Hu��_Procrustes distanceƥ��/������/1_1/result.txt");
//	// �������زĽ�������ƥ�䣬�۲���ƥ����
//	int count = 0;
//	for (int i = 0; i < 45; i++)
//	{
//		int best_index = MatchRank(points[i]);
//		if (i < 10)
//		{
//			cout << "����" << i << "��C1��        ƥ�䵽        ����" << best_index << "��C1��" << endl;
//		}
//		else
//		{
//			cout << "����" << i << "��C1��       ƥ�䵽        ����" << best_index << "��C1��" << endl;
//		}
//		outfile << i << "	<<	" << best_index << endl;
//
//
//		// ������ƥ�����������
//		// ��������������ƽ�Ƶ�һ��
//		// i����������
//		int x_i = points[i][12];
//		int y_i = points[i][1];
//		// j����������
//		int x_j = points[best_index][12];
//		int y_j = points[best_index][1];
//		// �ƶ�����
//		int x_move = x_j - x_i;
//		int y_move = y_j - y_i;
//		// ƽ��
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
//			// ԭͼΪ��ɫ
//			circle(picture, Point(points[i][j * 2], points[i][j * 2 + 1]), 2, CV_RGB(0, 0, 0), 2, 8, 0);
//			// ƥ��ͼΪ��ɫ
//			circle(picture, Point(points_j[j * 2], points_j[j * 2 + 1]), 2, CV_RGB(255, 0, 0), 2, 8, 0);
//		}
//
//		// ���������Ա�ͼ��
//		string pic_name = "E:/������/΢��/��״ƥ��/Hu��_Procrustes distanceƥ��/������/1_1/";
//		stringstream ss;
//		string str;
//		ss << i;
//		ss >> str;
//		pic_name.append(str + ".jpg");
//		imwrite(pic_name, picture);
//
//		// imshow("����" + i, picture);
//	}
//
//	outfile.close();
//
//	waitKey(0);
//	system("pause");
//
//	return 0;
//}


// C2��C1ƥ��
int main()
{
	// ���ı��ж���������м���
	ifstream infile;
	infile.open("E:/������/΢��/��״ƥ��/�ز�/����ͼƬ/facePoint.txt");
	for (int i = 0; i < 45; i++)
	{
		points[i] = new int[32];
		for (int j = 0; j < 32; j++)
		{
			infile >> points[i][j];
		}
	}
	infile.close();
	infile.open("E:/������/΢��/��״ƥ��/�ز�/����ͼƬ/facePointNew.txt");
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
	outfile.open("E:/������/΢��/��״ƥ��/Hu��_Procrustes distanceƥ��/������/1_2/result.txt");
	// �������زĽ�������ƥ�䣬�۲���ƥ����
	int count = 0;
	for (int i = 0; i < 45; i++)
	{
		int best_index = MatchRank(points_new[i]);
		if (i < 10)
		{
			cout << "����" << i << "��C2��        ƥ�䵽        ����" << best_index << "��C1��" << endl;
		}
		else
		{
			cout << "����" << i << "��C2��       ƥ�䵽        ����" << best_index << "��C1��" << endl;
		}
		outfile << i << "	<<	" << best_index << endl;


		// ������ƥ�����������
		// ��������������ƽ�Ƶ�һ��
		// i����������
		int x_i = points_new[i][12];
		int y_i = points_new[i][1];
		// j����������
		int x_j = points[best_index][12];
		int y_j = points[best_index][1];
		// �ƶ�����
		int x_move = x_j - x_i;
		int y_move = y_j - y_i;
		// ƽ��
		int points_j[32];
		for (int j = 0; j < 16; j++)
		{
			points_j[j * 2] = points[best_index][j * 2] - x_move;
			points_j[j * 2 + 1] = points[best_index][j * 2 + 1] - y_move;
		}

		Mat picture(img_width, img_height, CV_8UC3, Scalar(255, 255, 255));
		for (int j = 0; j < 16; j++)
		{
			// ԭͼΪ��ɫ
			circle(picture, Point(points_new[i][j * 2], points_new[i][j * 2 + 1]), 2, CV_RGB(0, 0, 0), 2, 8, 0);
			// ƥ��ͼΪ��ɫ
			circle(picture, Point(points_j[j * 2], points_j[j * 2 + 1]), 2, CV_RGB(255, 0, 0), 2, 8, 0);
		}

		// ���������Ա�ͼ��
		string pic_name = "E:/������/΢��/��״ƥ��/Hu��_Procrustes distanceƥ��/������/1_2/";
		stringstream ss;
		string str;
		ss << i;
		ss >> str;
		pic_name.append(str + ".jpg");

		// ��ͼƬ����ʾ���
		putText(picture, str, Point(400, 150), CV_FONT_HERSHEY_DUPLEX, 5.0f, CV_RGB(0, 0, 0));
		imwrite(pic_name, picture);

		// imshow("����" + i, picture);
		if (i == best_index)
		{
			count++;
		}
	}

	double rate = (double)count / 45;
	cout << endl << "ƥ����ȷ��Ϊ" << rate * 100 << "%" << endl << endl;

	outfile.close();

	waitKey(0);
	system("pause");

	return 0;
}