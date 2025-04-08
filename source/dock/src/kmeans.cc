#include"kmeans.h"
#include <vector>
#include <iostream>
#include <random>
#include <string.h>



void kmeans_init_centers(double * center, double * point,int point_number , int K){
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
	// std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::default_random_engine gen(42);
	std::uniform_int_distribution<int> distribution(0, point_number - 1);
    // 获得随机位置
	int id = distribution(gen);
    // 得到第一个聚类点
    for(int i = 0; i< K; i++){
        center[i*2] = point[id*2];
        center[i*2+1] = point[id*2+1];

    }
    // std::cout<<111<<std::endl;
    // 存放每个点到中心的最短距离
    double * floatIt  = new double [point_number];
    double sum, min_distance;
    // 初始化center
    // 从第二个点开始初始化
    for(int k = 1; k < K; k++){
        sum = 0; 
        // double * floatIt = &nearestDis[0];
        for(int p = 0; p<point_number; p++){
            // nearest center
            double * current_point = &point[p*2];
            min_distance = std::numeric_limits<double>::max();

            int k_id = -1;
            double dis;
            // 对已经确定的中心点遍历每一个点，得到聚类中心点距离
            for(int t = 0; t<k; t++){
                double * current_center = &center[t*2];
                dis = sqrt((current_point[0] - current_center[0])*(current_point[0] - current_center[0]) + (current_point[1] - current_center[1])*(current_point[1] - current_center[1]));
                // calc distance
                if(dis<min_distance){
                    min_distance = dis;
			        k_id = t;
                }

            }
            sum+=min_distance;
            floatIt[p] = min_distance;
        }
        // 得到总和和每一个点的距离已经确定的中心点的最小距离。
        std::random_device rd;  //Will be used to obtain a seed for the random number engine
		std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
		std::uniform_real_distribution<float> distribution(0.0, 1.0);
        double probability = distribution(gen);
        sum = sum*probability;
        for(int p = 0; p<point_number; p++){
            sum = sum - floatIt[p];
            if(sum>0) continue;
            center[k*2] = point[p*2];
            center[k*2+1] = point[p*2+1];
            break;
        }
        
    }
    delete [] floatIt;
}


void run_kmeans(int * group, double * points,double * centers, int max_iter, int point_number , int K){
    double * old_center = new double[K*2];
    for(int iter = 0; iter<max_iter; iter++){
        memcpy(old_center, centers, sizeof(double)*K*2);
        for(int i = 0; i<point_number; i++){
            double * current_point = & points[i*2];
            double minDistance = std::numeric_limits<float>::max();
            int k_id = -1;
            double dis;
            for(int k = 0; k<K; k++){
                double * current_center = &centers[k*2];
                dis = Distance(current_center, current_point);
                if(dis<minDistance){
                    minDistance = dis;
                    k_id = k;
                }
            }
            group[i] = k_id;

        }
        double * new_center = new double[K*2];
        int * count = new int[K];
        memset(new_center, 0.0, sizeof(double)*K*2);
        memset(count,0,sizeof(int)*K);
        for(int p = 0; p<point_number; p++){
            // 计算每个组的距离总和与元素总和
            int current_group = group[p];
            new_center[current_group*2] += points[p*2]; 
            new_center[current_group*2+1] += points[p*2+1]; 
            count[current_group]++;
        }
        for(int i = 0; i<K;i++){
            // 得到中心的平均值
            centers[i*2] = new_center[i*2] / (1.0*count[i]);
            centers[i*2+1] = new_center[i*2+1] / (1.0*count[i]);
        }

        double sum = 0;
        for(int k = 0; k<K; k++){
            sum += Distance(&centers[k*2], &old_center[k*2]);
        }
        std::cout << "iteration "<< iter<<" sum " << sum << std::endl;
        delete [] new_center;
        delete [] count;
		
        
    }
    delete [] old_center;

}





void kmeans_init_centers3(double * center, double * point,int point_number , int K, int seed){
    // std::random_device rd;  //Will be used to obtain a seed for the random number engine
    
	std::mt19937 gen(seed); //Standard mersenne_twister_engine seeded with rd()
	std::uniform_int_distribution<int> distribution(0, point_number - 1);
    // 获得随机位置
	int id = distribution(gen);
    // 得到第一个聚类点
    for(int i = 0; i< K; i++){
        center[i*3] = point[id*3];
        center[i*3+1] = point[id*3+1];
        center[i*3+2] = point[id*3+2];

    }
    // std::cout<<111<<std::endl;
    // 存放每个点到中心的最短距离
    double * floatIt  = new double [point_number];
    double sum, min_distance;
    // 初始化center
    // 从第二个点开始初始化
    for(int k = 1; k < K; k++){
        sum = 0; 
        // double * floatIt = &nearestDis[0];
        for(int p = 0; p<point_number; p++){
            // nearest center
            double * current_point = &point[p*3];
            min_distance = std::numeric_limits<double>::max();

            int k_id = -1;
            double dis;
            // 对已经确定的中心点遍历每一个点，得到聚类中心点距离
            for(int t = 0; t<k; t++){
                double * current_center = &center[t*3];
                dis = Distance3(current_point, current_center);
                // sqrt((current_point[0] - current_center[0])*(current_point[0] - current_center[0]) + (current_point[1] - current_center[1])*(current_point[1] - current_center[1]));
                // calc distance
                if(dis<min_distance){
                    min_distance = dis;
			        k_id = t;
                }

            }
            sum+=min_distance;
            floatIt[p] = min_distance;
        }
        // 得到总和和每一个点的距离已经确定的中心点的最小距离。
        std::random_device rd;  //Will be used to obtain a seed for the random number engine
		// std::mt19937 gen(seed); //Standard mersenne_twister_engine seeded with rd()
		std::uniform_real_distribution<float> distribution(0.0, 1.0);
        double probability = distribution(gen);
        sum = sum*probability;
        for(int p = 0; p<point_number; p++){
            sum = sum - floatIt[p];
            if(sum>0) continue;
            center[k*3] = point[p*3];
            center[k*3+1] = point[p*3+1];
            center[k*3+2] = point[p*3+2];
            break;
        }
        
    }
    delete [] floatIt;
}


void run_kmeans3(int * group, double * points,double * centers, int max_iter, int point_number , int K){
    double * old_center = new double[K*3];
    for(int iter = 0; iter<max_iter; iter++){
        memcpy(old_center, centers, sizeof(double)*K*3);
        for(int i = 0; i<point_number; i++){
            double * current_point = & points[i*3];
            double minDistance = std::numeric_limits<float>::max();
            int k_id = -1;
            double dis;
            for(int k = 0; k<K; k++){
                double * current_center = &centers[k*3];
                dis = Distance3(current_center, current_point);
                if(dis<minDistance){
                    minDistance = dis;
                    k_id = k;
                }
            }
            group[i] = k_id;

        }
        double * new_center = new double[K*3];
        int * count = new int[K];
        memset(new_center, 0.0, sizeof(double)*K*3);
        memset(count,0,sizeof(int)*K);
        for(int p = 0; p<point_number; p++){
            // 计算每个组的距离总和与元素总和
            int current_group = group[p];
            new_center[current_group*3] += points[p*3]; 
            new_center[current_group*3+1] += points[p*3+1];
            new_center[current_group*3+2] += points[p*3+2]; 
            count[current_group]++;
        }
        for(int i = 0; i<K;i++){
            // 得到中心的平均值
            centers[i*3] = new_center[i*3] / (1.0*count[i]);
            centers[i*3+1] = new_center[i*3+1] / (1.0*count[i]);
            centers[i*3+2] = new_center[i*3+2] / (1.0*count[i]);
        }

        // double sum = 0;
        // for(int k = 0; k<K; k++){
        //     sum += Distance3(&centers[k*3], &old_center[k*3]);
        // }
        // std::cout << "iteration "<< iter<<" sum " << sum << std::endl;
        delete [] new_center;
        delete [] count;
		
        
    }
    delete [] old_center;

}