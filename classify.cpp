//
//  classify.cpp
//  transfer-learning
//
//  Created by Kushashwa Ravi Shrimali on 15/08/19.
//  Copyright © 2019 Kushashwa Ravi Shrimali. All rights reserved.
//

#include <iostream>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <torch/script.h>

//#include "model.h"

int main(int arc, char** argv)
{
    std::string loc = argv[1];
    
    // Load image with OpenCV.
    cv::Mat img = cv::imread(loc);
    cv::resize(img, img, cv::Size(224, 224), cv::INTER_CUBIC);
    // Convert the image and label to a tensor.
    torch::Tensor img_tensor = torch::from_blob(img.data, {1, img.rows, img.cols, 3}, torch::kByte);
    img_tensor = img_tensor.permute({0, 3, 1, 2}); // convert to CxHxW
    img_tensor = img_tensor.to(torch::kF32);
    
    // Load the model.
    torch::jit::script::Module model;
    model = torch::jit::load(argv[2]);
    
    std::cout << "Model loaded" << std::endl;
    // Predict the probabilities for the classes.
    std::vector<torch::jit::IValue> input;
    input.push_back(img_tensor);
    torch::Tensor prob = model.forward(input).toTensor();
//    torch::Tensor prob = torch::exp(log_prob);
    
    std::cout << "Probability of being cat: " << *(prob.data<float>())*100. << ", of being dog: " << *(prob.data<float>() + 1)*100. << std::endl;
    
    return 0;
}
