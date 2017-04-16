#pragma once
#include "traincascade\traincascade_features.h"
class SHOGfeatures :
    public CvFeatureEvaluator
{
public:
    SHOGfeatures();
    ~SHOGfeatures();
};

