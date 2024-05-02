#include <dirent.h>
#include "common.h"
#include <stdlib.h>
#include <stdio.h>
#include <cstdlib>
//#include <Windows.h>

using namespace std;
using namespace cv;


int SaveDescrBinary(string path, const double* descr, long descr_size)
{
	FILE* f = fopen(path.c_str(), "wb");
	if (f == NULL)
		return -1;
	
	fwrite(descr, sizeof(double), descr_size, f);
	fclose(f);

	string t_path = path;
	size_t idx = path.find_last_of('.');						
	t_path.insert(idx, "_info");
	f = fopen(t_path.c_str(), "w");
	//STORE DESCRIPTOR SIZE
	fprintf(f, "%d", descr_size);	
	fclose(f);

	return 1;
}

int SaveDescrBinary(string path, float* descr, long descr_size)
{
	FILE* f = fopen(path.c_str(), "wb");
	if (f == NULL)
		return -1;
	
	fwrite(descr, sizeof(float), descr_size, f);
	fclose(f);

	string t_path = path;
	const int idx = path.find_last_of('.');						
	t_path.insert(idx, "_info");
	f = fopen(t_path.c_str(), "wb");
	//STORE DESCRIPTOR SIZE
	fprintf(f, "%d\n", descr_size);	
	fclose(f);

	return 1;
}

int LoadDescrBinary(string path, double* descr, long size)
{
	FILE* f = fopen(path.c_str(), "rb");
	if (f == NULL)
		return -1;
	
	fread(descr , sizeof(double), size, f);

	fclose(f);

	return 1;
}

int LoadDescrBinary(string path, float* descr, long size)
{
	FILE* f = fopen(path.c_str(), "rb");
	if (f == NULL)
		return -1;
	
	fread(descr , sizeof(float), size, f);

	fclose(f);

	return 1;
}

int LoadDescrBinaryInfo(string path, long *size)
{
	FILE* f = fopen(path.c_str(), "rb");
	if(f == NULL)
		return - 1;
	long temp = 0;
	fscanf(f, "%d\n", &temp);

	*size = temp;
	/*std::ifstream f(path);
	if(!f.is_open())
		return -1;
	
	string s;
	f >> s ;
	*size = stod(s);*/

	return 1;
}

int LoadTextFile(string path, vector<string> &output)
{

	char str[256];
    int p1, p2, p3, p4;
    char p5[16];
    FILE *fp;
    int i;
	fp=fopen(path.c_str(), "r");
    while(fgets(str, sizeof(str), fp)!=NULL)
	{
		int l = strlen(str);
		if(str[l - 1] == '\n' || str[l - 1] == '\r')
			str[l - 1] = '\0';
		string line(str);		
		output.push_back(line);
    }
    fclose(fp);

	/*ifstream in(path);
	if (in.fail())
		return -1;*/

	//FILE* f;
	//fopen_s(&f, path.c_str(), "r");
	//
	//char* buffer = 0;
	//buffer = (char *)malloc(sizeof(char)*100000);

	////temp = (char *)malloc(sizeof(char)* 256);

	//int i = 0, j = 0;

	//char* FIN = fgets(buffer, 100000, f);
	//int countAdj = 0;
	//std::string line;
	//while (FIN != 0)
	//{
	//	j = 0;
	//	i = 0;
	//	char* temp = (char *)malloc(sizeof(char)* 256);
	//	while (buffer[i] != '\r' && buffer[i] != '\n')
	//	{
	//		temp[j] = buffer[i];
	//		i++;
	//		j++;
	//	}			
	//	temp = (char*)realloc(temp, sizeof(char)*j);
	//	temp[j] = '\0';	
	//	line = string(temp);
	//	output.push_back(temp);
	//	FIN = fgets(buffer, 100000, f);
	//}
	//free(buffer);
	//std::string line;
	//while (std::getline(in, line))
	//{
	//	std::istringstream iss(line);
	//	int a, b;
	//	if (in >> line)
	//	{
	//		output.push_back(line);
	//	} // error

	//	// process pair (a,b)
	//}

	return 1;
}

int SaveDescrAsText(string path, vector<float> descr, int dim_desc)
{
	int d = 1;
	std::ofstream fout(path);
	if (fout.fail()){
		cout << "ERROR opening file " << path << endl;
		return -1;
	}

	for (int f = 0 ; f < descr.size() ; f++)
	{
		fout << descr[f] << " "   ;	
		if (f == (d * dim_desc - 1))
		{
			fout << endl;
			d++;					
		}				
	}

	return 1;
}

int SaveDescrAsText(string path, vector<float> descr)
{
	std::ofstream fout(path);
	if (fout.fail()){
		cout << "ERROR opening file " << path << endl;
		return -1;
	}

	for (int f = 0 ; f < descr.size() ; f++)
		fout << descr[f] << endl;	

	return 1;
}

int loadDBInfo(string path, int* num_images)
{
	FILE* f = fopen(path.c_str(), "r");
	if (f == NULL)
		return -1;
	
	fscanf(f, "%d\n", num_images);
	fclose(f);

	return 1;
}

int SaveDBInfo(string path, int num_images)
{
	FILE* f = fopen(path.c_str(), "wb");
	if (f == NULL)
		return -1;
	
	fprintf(f, "%d\n", num_images);
	fclose(f);

	return 1;
}

int SaveDescrChunkBinary(string path, float* descr, long descr_size)
{
	FILE* f = fopen(path.c_str(), "wb");
	if (f == NULL)
		return -1;
	
	fwrite(descr, sizeof(float), descr_size, f);
	fclose(f);

	//string t_path = path;
	//size_t idx = path.find_last_of('.');						
	//t_path.insert(idx, "_info");
	//f = fopen(t_path.c_str(), "w");
	////STORE DESCRIPTOR SIZE
	//fprintf(f, "%d\n", descr_size);	
	//fprintf(f, "%d\n", offset);
	//fclose(f);

	return 1;
}

double* GetLabels(vector<string> image_url, const char* positive_class)
{	
	double* labels = (double*)malloc(sizeof(double)*image_url.size());
	for (int i = 0; i < image_url.size(); i++)
		labels[i] = (strstr(image_url[i].c_str(), positive_class) != NULL) ? 1.0 : -1.0;
	return labels;
}

double* GetLabels(vector<string> image_url, const char* positive_class, int* fake_count, int* live_count)
{	
	double* labels = (double*)malloc(sizeof(double)*image_url.size());
	int live =0, fake = 0;
	for (int i = 0; i < image_url.size(); i++)
	{
		if(strstr(image_url[i].c_str(), positive_class) != NULL)
		{
			labels[i] =  1.0;
			live++;
		}
		else
		{
			fake++;
			labels[i] =  -1.0;
		}		
	}

	*fake_count = fake;
	*live_count = live;
	return labels;
}

void copyFile(const std::string& fileNameFrom, const std::string& fileNameTo)
{
     assert(is_file_exist(fileNameFrom.c_str()));
     std::ifstream in (fileNameFrom.c_str());
     std::ofstream out (fileNameTo.c_str());
     out << in.rdbuf();
     out.close();
     in.close();
}

Mat readImage(string image_path)
{
	Mat img_i = imread(image_path, CV_LOAD_IMAGE_GRAYSCALE);
	if(img_i.rows > 480)
	{
		Mat temp;
		resize(img_i, temp, Size(img_i.cols, 480), 0, 0, CV_INTER_CUBIC);
		img_i = temp;
	}
	if(img_i.rows > 400)
	{
		Mat temp;
		resize(img_i, temp, Size(img_i.cols*0.78, img_i.rows*0.78), 0.0, 0.0, CV_INTER_CUBIC);
		img_i = temp;
	}
	return img_i;
}

int load_model(string path, float* model[], float bias[], int* model_size, int nr_classes)
{
	string t_path = path;
	size_t idx = path.find_last_of('.');
	t_path.insert(idx, "_info");
	FILE* f = fopen(t_path.c_str(), "rb");
	//STORE DESCRIPTOR SIZE
	int s, no_classes= 0;
	fscanf(f, "%d\n", &s);	
	fscanf(f, "%d\n", &no_classes);
	fclose(f);
	*model_size = s;
	f = fopen(path.c_str(), "rb");
	if (f == NULL)
		return -1;

	for(int i = 0; i < nr_classes; i++)
	{
		model[i] = (float*)malloc(sizeof(float)*s);
		fread(model[i], sizeof(float), s, f);
		fread(&bias[i], sizeof(float), 1, f);
	}

	fclose(f);

	return 1;
}

int save_model(string path, float* model[], float* bias, int model_size, int nr_classes)
{
	FILE* f = fopen(path.c_str(), "wb");
	if (f == NULL)
		return -1;

	for(int i = 0; i < nr_classes; i++)
	{
		fwrite(model[i], sizeof(float), model_size, f);
		fwrite(&(bias[i]), sizeof(float), 1, f);
	}
	
	fclose(f);

	string t_path = path;
	size_t idx = path.find_last_of('.');
	t_path.insert(idx, "_info");
	f = fopen(t_path.c_str(), "w");
	//STORE DESCRIPTOR SIZE
	fprintf(f, "%d\n", model_size);
	fprintf(f, "%d\n", nr_classes);
	fclose(f);

	return 1;
}

bool Extencion(string a, string exten)
{
	bool contine = false;
	int length_a = a.length();
	int length_ex = exten.length();

	int index = length_a - length_ex;
	int i = 0;
	for (; index<length_a; index++)
	{
		if (a[index] != exten[i])
		{
			return false;

		}
		i++;
	}
	return true;
}

inline std::string get_extension(const std::string &file_path)
{
	string ext(strrchr(file_path.c_str(), '.'));
	return ext;
}

int get_files(const string folder_path, const string extension, vector<string> &files, int max_number, const bool deeper_search)
{
	DIR *dir;
	struct dirent *ent;
	
	if ((dir = opendir(folder_path.c_str())) != NULL)
	{
		while ((ent = readdir(dir)) != NULL && max_number > 0)
		{
			if (ent->d_type == DT_DIR)
			{
				if (strcmp(ent->d_name, ".") && strcmp(ent->d_name, "..") && deeper_search)
					get_files(folder_path + "/" + ent->d_name, extension, files, max_number, deeper_search);
		    }
			else
			{
				string current_ext = get_extension(ent->d_name);
				if (extension == current_ext)
				{
					files.push_back(folder_path + "/" + ent->d_name);
					max_number--;
				}
			}
		}
		closedir(dir);
	}
	if (files.size() == 0)
		//cout << "ERROR: Cannot open folder path " << folder_path << endl;
		return -1;

	return 1;
}

bool is_file_exist(const char *fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}

string to_string_(int value)
{
    char buf[128];
    sprintf(buf, "%d", value);
    return string(buf);
}


