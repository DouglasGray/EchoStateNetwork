//
//  Attributes.h
//  ESN
//
//  Created by admin on 01/10/2014.
//  Copyright (c) 2014 AerophobicEmpurplement. All rights reserved.
//

#ifndef ESN_Attributes_h
#define ESN_Attributes_h

/*--- Header file containing useful structures ---*/

// Enumerations for the possible layer activation functions
enum act_fcts
{
    TANH,
    IDENTITY,
    SIGMOID,
};

// Enumerations for training algorithms: pseudo-inverse and
// Tikhinov regularisation.
// Wiener-Hopf has not been included since it is a special case of
// Tikhinov (beta = 0).
enum training_algos
{
    PINV,
    TIKHINOV,
};

// Structure containing network parameters and default initialisations
struct Attributes {
    int ninput = -1;
    int ninternal = -1;
    int noutput = -1;
    int resf = TANH;
    int outf = TANH;
    int solution_method = TIKHINOV;
    int washout = 100;
    
    double radius = -1;
    double alpha = 0.5;
    double beta = 0.0;
    double density = -1;
    
    double input_shift = 0.0;
    double input_scaling = 1.0;
    double teacher_shift = 0.0;
    double teacher_scaling = 1.0;
    double feedback_scaling = 0.0;
    
    bool noise = false;
};


#endif
