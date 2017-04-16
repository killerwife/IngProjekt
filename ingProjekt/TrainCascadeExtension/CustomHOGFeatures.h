#pragma once
#include "../traincascade/traincascade_features.h"
class CustomHOGFeatures :
    public CvFeatureEvaluator
{
public:
    CustomHOGFeatures();
    virtual ~CustomHOGFeatures();
};

