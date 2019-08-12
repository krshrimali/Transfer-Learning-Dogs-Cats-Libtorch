//
//  main.cpp
//  transfer-learning
//
//  Created by Kushashwa Ravi Shrimali on 12/08/19.
//  Copyright © 2019 Kushashwa Ravi Shrimali. All rights reserved.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <dirent.h>
#include <torch/script.h>

torch::Tensor read_data(std::string location) {
    cv::Mat img = cv::imread(location, 1);
    cv::resize(img, img, cv::Size(224, 224), cv::INTER_CUBIC);
    torch::Tensor img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte);
    img_tensor = img_tensor.permute({2, 0, 1});
    return img_tensor.clone();
}

torch::Tensor read_label(int label) {
    torch::Tensor label_tensor = torch::full({1}, label);
    return label_tensor.clone();
}

std::vector<torch::Tensor> process_images(std::vector<std::string> list_images) {
    std::vector<torch::Tensor> states;
    for(std::vector<std::string>::iterator it = list_images.begin(); it != list_images.end(); ++it) {
        torch::Tensor img = read_data(*it);
        states.push_back(img);
    }
    return states;
}

std::vector<torch::Tensor> process_labels(std::vector<int> list_labels) {
    std::vector<torch::Tensor> labels;
    for(std::vector<int>::iterator it = list_labels.begin(); it != list_labels.end(); ++it) {
        torch::Tensor label = read_label(*it);
        labels.push_back(label);
    }
    return labels;
}

std::pair<std::vector<std::string>,std::vector<int>> load_data_from_folder(std::vector<std::string> folders_name) {
    std::vector<std::string> list_images;
    std::vector<int> list_labels;
    int label = 0;
    for(auto const& value: folders_name) {
        std::string base_name = value + "/";
        // cout << "Reading from: " << base_name << endl;
        DIR* dir;
        struct dirent *ent;
        if((dir = opendir(base_name.c_str())) != NULL) {
            while((ent = readdir(dir)) != NULL) {
                std::string filename = ent->d_name;
                if(filename.length() > 4 && filename.substr(filename.length() - 3) == "jpg") {
                    // cout << base_name + ent->d_name << endl;
                    // cv::Mat temp = cv::imread(base_name + "/" + ent->d_name, 1);
                    list_images.push_back(base_name + ent->d_name);
                    list_labels.push_back(label);
                }
                
            }
            closedir(dir);
        } else {
            std::cout << "Could not open directory" << std::endl;
            // return EXIT_FAILURE;
        }
        label += 1;
    }
    return std::make_pair(list_images, list_labels);
}

class CustomDataset : public torch::data::Dataset<CustomDataset> {
private:
    /* data */
    // Should be 2 tensors
    std::vector<torch::Tensor> states, labels;
public:
    CustomDataset(std::vector<std::string> list_images, std::vector<int> list_labels) {
        states = process_images(list_images);
        labels = process_labels(list_labels);
    };
    
    torch::data::Example<> get(size_t index) override {
        /* This should return {torch::Tensor, torch::Tensor} */
        torch::Tensor sample_img = states.at(index);
        torch::Tensor sample_label = labels.at(index);
        return {sample_img.clone(), sample_label.clone()};
    };
    
    torch::optional<size_t> size() const override {
        return states.size();
    };
};

template<typename Dataloader>
void train(torch::jit::script::Module net, torch::nn::Linear lin, Dataloader& data_loader, torch::optim::Optimizer& optimizer, size_t dataset_size, int epoch) {
    /*
     This function trains the network on our data loader using optimizer for given number of epochs.
     
     Parameters
     ==================
     ConvNet& net: Network struct
     DataLoader& data_loader: Training data loader
     torch::optim::Optimizer& optimizer: Optimizer like Adam, SGD etc.
     size_t dataset_size: Size of training dataset
     int epoch: Number of epoch for training
     */
    
    net.train();
    
    size_t batch_index = 0;
    float mse = 0;
    float Acc = 0.0;
    
    for(auto& batch: *data_loader) {
        auto data = batch.data;
        auto target = batch.target.squeeze();
        
        // Should be of length: batch_size
        data = data.to(torch::kF32);
        target = target.to(torch::kInt64);
        
        std::vector<torch::jit::IValue> input;
        input.push_back(data);
        
        auto output = net.forward(input).toTensor().squeeze();
        // For transfer learning
        output = lin(output);
        auto loss = torch::nll_loss(output, target);
        
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        
        auto acc = output.argmax(1).eq(target).sum();
        Acc += acc.template item<float>();
        mse += loss.template item<float>();
        
        batch_index += 1;
    }
    mse = mse/float(batch_index); // Take mean of loss
    
    std::cout << "Epoch: " << epoch << ", " << "Accuracy: " << Acc/dataset_size << ", " << "MSE: " << mse << std::endl;
//    torch::save(net, "best_model_try.pt");
}

template<typename Dataloader>
void test(torch::jit::script::Module network, torch::nn::Linear lin, Dataloader& loader, size_t data_size) {
    // size_t batch_index = 0;
    
    network.eval();
        
    float Loss = 0, Acc = 0;
    
    for (const auto& batch : *loader) {
        auto data = batch.data;
        auto targets = batch.target.view({-1});
        
        data = data.to(torch::kF32);
        targets = targets.to(torch::kInt64);
        std::vector<torch::jit::IValue> input;
        input.push_back(data);
        auto output = network.forward(input).toTensor();
        output = lin(output);
        // std::cout << output << std::endl;
        auto loss = torch::nll_loss(output, targets);
        auto acc = output.argmax(1).eq(targets).sum();
        
        Loss += loss.template item<float>();
        Acc += acc.template item<float>();
    }
    std::cout << "Total Loss: " << Loss << std::endl;
    std::cout << "Total Accuracy: " << Acc << std::endl;
    std::cout << "Test Loss: " << Loss/data_size << ", Acc:" << Acc/data_size << std::endl;
}

int main(int argc, const char * argv[]) {
    // insert code here...
    std::cout << "Hello, World!\n";
    
    std::string cats_name = "/Users/krshrimali/Documents/krshrimali-blogs/dataset/train/cat_test";
    std::string dogs_name = "/Users/krshrimali/Documents/krshrimali-blogs/dataset/train/dog_test";
    
    std::vector<std::string> folders_name;
    folders_name.push_back(cats_name);
    folders_name.push_back(dogs_name);
    
    std::pair<std::vector<std::string>, std::vector<int>> pair_images_labels = load_data_from_folder(folders_name);
    
    std::cout << "Data loaded" << std::endl;
    
    std::vector<std::string> list_images = pair_images_labels.first;
    std::vector<int> list_labels = pair_images_labels.second;
    
    auto custom_dataset = CustomDataset(list_images, list_labels).map(torch::data::transforms::Stack<>());
    
    std::cout << "Dataset initialized" << std::endl;
    
    torch::jit::script::Module module;
    module = torch::jit::load(argv[1]);
    
    std::cout << "Problem here" << std::endl;
    
    // Resourcee: https://discuss.pytorch.org/t/how-to-load-the-prebuilt-resnet-models-or-any-other-prebuilt-models/40269/8
    
    torch::nn::Linear lin(512 , 2); // the last layer of resnet, which you want to replace, has dimensions 512x1000
    torch::optim::Adam opt(lin->parameters(), torch::optim::AdamOptions(1e-3 /*learning rate*/));
    
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset),64);
    std::cout << "Size is: " << custom_dataset.size().value() << std::endl;
    
//    torch::optim::Adam optimizer(module.get_parameters(), torch::optim::AdamOptions(1e-3));
    
    for(int i = 0; i < 10; i++) {
        train(module, lin, data_loader, opt, custom_dataset.size().value(), i);
        test(module, lin, data_loader, custom_dataset.size().value());
    }
    return 0;
}
