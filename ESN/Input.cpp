//
//  Input.cpp
//  ESN
//
//  Created by admin on 01/10/2014.
//  Copyright (c) 2014 AerophobicEmpurplement. All rights reserved.
//

#include "Input.h"
#include "Attributes.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <getopt.h>
#include <cstring>

// Various options for network configuration
static struct option long_options[] = {
    {"config_file", required_argument, 0, 'a'},
    {"input_units", required_argument, 0, 'b'},
    {"internal_units", required_argument, 0, 'c'},
    {"output_units", required_argument, 0, 'd'},
    {"leaking_rate", required_argument, 0, 'e'},
    {"regularisation", required_argument, 0, 'f'},
    {"output_act", required_argument, 0, 'g'},
    {"reservoir_act", required_argument, 0, 'h'},
    {"density", required_argument, 0, 'i'},
    {"noise", required_argument, 0, 'j'},
    {"spectral_radius", required_argument, 0, 'k'},
    {"washout", required_argument, 0, 'l'},
    {"solution_method", required_argument, 0, 'm'},
    {"input_shift", required_argument, 0, 'n'},
    {"input_scaling", required_argument, 0, 'o'},
    {"teacher_shift", required_argument, 0, 'p'},
    {"teacher_scaling", required_argument, 0, 'q'},
    {"feedback_scaling", required_argument, 0, 'r'},
    {0, 0, 0, 0}
};

// Constructor
Input::Input() {};

// Simply cycles through the given arguments and checks if any of the file configurations have been overwritten
void Input::parse(int argc, char **argv, Attributes &atts)
{
    int c = 0;
    int long_index = 0;
    char *file = NULL;
    
    // Cycle through command line input and gather the relevant arguments
    while ((c = getopt_long(argc, argv, "a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:", long_options, &long_index)) != -1)
    {
        switch (c)
        {
            case 'a':
                file = optarg;
                parse_file(file, atts);
                break;
            case 'b':
                atts.ninput = atoi(optarg);
                break;
            case 'c':
                atts.ninternal = atoi(optarg);
                break;
            case 'd':
                atts.noutput = atoi(optarg);
                break;
            case 'e':
                atts.alpha = atof(optarg);
                break;
            case 'f':
                atts.beta = atof(optarg);
                break;
            case 'g':
                if (std::strcmp(optarg,"tanh"))
                    atts.outf = TANH;
                else if (std::strcmp(optarg, "identity"))
                    atts.outf = IDENTITY;
                else if (std::strcmp(optarg, "sigmoid"))
                    atts.outf = SIGMOID;
                else
                {
                    std::cout << "Parser::parse_input: Output activation function not recognised" << optarg << std::endl;
                    exit(EXIT_FAILURE);
                }
                break;
            case 'h':
                if (strcmp(optarg,"tanh"))
                    atts.resf = TANH;
                else if (strcmp(optarg, "identity"))
                    atts.resf = IDENTITY;
                else if (strcmp(optarg, "sigmoid"))
                    atts.resf = SIGMOID;
                else
                {
                    std::cout << "Parser::parse_input: Reservoir activation function not recognised" << optarg << std::endl;
                    exit(EXIT_FAILURE);
                }
                break;
            case 'i':
                atts.density = atof(optarg);
                break;
            case 'j':
                atts.noise = atof(optarg);
                break;
            case 'k':
                atts.radius = atof(optarg);
                break;
            case 'l':
                atts.washout = atoi(optarg);
                break;
            case 'm':
                if (strcmp(optarg,"pseudo_inverse"))
                    atts.solution_method = PINV;
                else if (strcmp(optarg, "tikhinov"))
                    atts.solution_method = TIKHINOV;
                else
                {
                    std::cout << "Parser::parse_input: Output weight solution method not recognised: " << optarg << std::endl;
                    exit(EXIT_FAILURE);
                }
                break;
            case 'n':
                atts.input_shift = atof(optarg);
                break;
            case 'o':
                atts.input_scaling = atof(optarg);
                break;
            case 'p':
                atts.teacher_shift = atof(optarg);
                break;
            case 'q':
                atts.teacher_scaling = atof(optarg);
                break;
            case 'r':
                atts.feedback_scaling = atof(optarg);
                break;
            default:
                std::cout << "Parser::parse_input: Unknown option: " << optarg << std::endl;
                break;
        }
    }
}

// Checks the file for network arguments
void Input::parse_file(char *filename, Attributes &atts)
{
    std::ifstream input(filename);
    std::string buff, opt, val;
    
    if (input.fail())
    {
        std::cout << "Input::parse_file: Could not open given input file" << std::endl;
        exit(EXIT_FAILURE);
    }
    
    while (!input.eof())
    {
        // Get the attribute
        std::getline(input, buff);
        
        // Put into stringstream to split up
        std::stringstream ss(buff);
        
        // Split into name/value pairs
        ss >> opt;
        ss >> val;
        
        if (!opt.compare("input_units"))
            atts.ninput = atoi(val.c_str());
        else if (!opt.compare("internal_units"))
            atts.ninternal = atoi(val.c_str());
        else if (!opt.compare("output_units"))
            atts.noutput = atoi(val.c_str());
        else if (!opt.compare("leaking_rate"))
            atts.alpha = atof(val.c_str());
        else if (!opt.compare("regularisation"))
            atts.beta = atof(val.c_str());
        else if (!opt.compare("output_act"))
        {
            if (!val.compare("tanh"))
                atts.outf = TANH;
            else if (!val.compare("identity"))
                atts.outf = IDENTITY;
            else if (!val.compare("sigmoid"))
                atts.outf = SIGMOID;
            else
            {
                std::cout << "Input::parse_file: Output activation function not recognised: " << opt << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        else if (!opt.compare("reservoir_act"))
        {
            if (!val.compare("tanh"))
                atts.resf = TANH;
            else if (!val.compare("identity"))
                atts.resf = IDENTITY;
            else if (!val.compare("sigmoid"))
                atts.resf = SIGMOID;
            else
            {
                std::cout << "Input::parse_file: Reservoir activation function not recognised: " << opt << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        else if (!opt.compare("density"))
            atts.density = atof(val.c_str());
        else if (!opt.compare("noise"))
            atts.noise = atof(val.c_str());
        else if (!opt.compare("spectral_radius"))
            atts.radius = atof(val.c_str());
        else if (!opt.compare("washout"))
            atts.washout = atoi(val.c_str());
        else if (!opt.compare("solution_method"))
        {
            if (!val.compare("pseudo_inv"))
                atts.solution_method = PINV;
            else if (!val.compare("tikhinov"))
                atts.solution_method = TIKHINOV;
            else
            {
                std::cout << "Input::parse_file: Output weight solution method not recognised: " << opt << std::endl;
                exit(EXIT_FAILURE);
            }
        }
        else if (!opt.compare("input_shift"))
            atts.input_shift = atof(val.c_str());
        else if (!opt.compare("input_scaling"))
            atts.input_scaling = atof(val.c_str());
        else if (!opt.compare("teacher_shift"))
            atts.teacher_shift = atof(val.c_str());
        else if (!opt.compare("teacher_scaling"))
            atts.teacher_scaling = atof(val.c_str());
        else if (!opt.compare("feedback_scaling"))
            atts.feedback_scaling = atof(val.c_str());
        else
        {
            std::cout << "Input::parse_file: Network configuration parameter not recognised: " << opt <<std::endl;
            exit(EXIT_FAILURE);
        }
    }
    input.close();
}
