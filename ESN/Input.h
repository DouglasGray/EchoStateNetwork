//
//  Input.h
//  ESN
//
//  Created by admin on 01/10/2014.
//  Copyright (c) 2014 AerophobicEmpurplement. All rights reserved.
//

#ifndef __ESN__Input__
#define __ESN__Input__

#include <iostream>
#include "Attributes.h"

class Input
{
public:
    Input();
    void parse(int argc, char** argv, Attributes &atts);
    
private:
    void parse_file(char *name, Attributes &att);
};


#endif /* defined(__ESN__Input__) */
