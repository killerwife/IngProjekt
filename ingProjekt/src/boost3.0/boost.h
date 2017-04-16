#include "precomp.hpp"
#include "opencv2/ml.hpp"

namespace CV
{
    class DTreesImplForBoostOwn : public cv::ml::DTreesImpl
    {
    public:
        DTreesImplForBoostOwn()
        {
            params.setCVFolds(0);
            params.setMaxDepth(1);
        }
        virtual ~DTreesImplForBoostOwn() {}

        bool isClassifier() const { return true; }

        void clear()
        {
            DTreesImpl::clear();
        }

        void startTraining(const cv::Ptr<cv::ml::TrainData>& trainData, int flags);
        void normalizeWeights();
        void endTraining();
        void scaleTree(int root, double scale);
        void calcValue(int nidx, const std::vector<int>& _sidx);
        bool train(const cv::Ptr<cv::ml::TrainData>& trainData, int flags);
        void updateWeightsAndTrim(int treeidx, std::vector<int>& sidx);
        float predictTrees(const cv::Range& range, const cv::Mat& sample, int flags0) const;
        void writeTrainingParams(cv::FileStorage& fs) const;
        void write(cv::FileStorage& fs) const;
        void readParams(const cv::FileNode& fn);
        void read(const cv::FileNode& fn);

        cv::ml::BoostTreeParams bparams;
        std::vector<double> sumResult;
    };

    class BoostOwn : public cv::ml::Boost
    {
    public:
        BoostOwn() {}
        virtual ~BoostOwn() {}

        CV_IMPL_PROPERTY(int, BoostType, impl.bparams.boostType)
            CV_IMPL_PROPERTY(int, WeakCount, impl.bparams.weakCount)
            CV_IMPL_PROPERTY(double, WeightTrimRate, impl.bparams.weightTrimRate)

            CV_WRAP_SAME_PROPERTY(int, MaxCategories, impl.params)
            CV_WRAP_SAME_PROPERTY(int, MaxDepth, impl.params)
            CV_WRAP_SAME_PROPERTY(int, MinSampleCount, impl.params)
            CV_WRAP_SAME_PROPERTY(int, CVFolds, impl.params)
            CV_WRAP_SAME_PROPERTY(bool, UseSurrogates, impl.params)
            CV_WRAP_SAME_PROPERTY(bool, Use1SERule, impl.params)
            CV_WRAP_SAME_PROPERTY(bool, TruncatePrunedTree, impl.params)
            CV_WRAP_SAME_PROPERTY(float, RegressionAccuracy, impl.params)
            CV_WRAP_SAME_PROPERTY_S(cv::Mat, Priors, impl.params)

            cv::String getDefaultName() const { return "opencv_ml_boost"; }

        bool train(const cv::Ptr<cv::ml::TrainData>& trainData, int flags = 0)
        {
            return impl.train(trainData, flags);
        }

        float predict(cv::InputArray samples, cv::OutputArray results, int flags = 0) const
        {
            return impl.predict(samples, results, flags);
        }

        void write(cv::FileStorage& fs) const
        {
            impl.write(fs);
        }

        void read(const cv::FileNode& fn)
        {
            impl.read(fn);
        }

        int getVarCount() const { return impl.getVarCount(); }

        bool isTrained() const { return impl.isTrained(); }
        bool isClassifier() const { return impl.isClassifier(); }

        const std::vector<int>& getRoots() const { return impl.getRoots(); }
        const std::vector<cv::ml::DTrees::Node>& getNodes() const { return impl.getNodes(); }
        const std::vector<cv::ml::DTrees::Split>& getSplits() const { return impl.getSplits(); }
        const std::vector<int>& getSubsets() const { return impl.getSubsets(); }

        static cv::Ptr<BoostOwn> create()
        {
            return cv::makePtr<BoostOwn>();
        }

        static cv::Ptr<BoostOwn> load(const cv::String& filepath, const cv::String& nodeName)
        {
            return cv::Algorithm::load<BoostOwn>(filepath, nodeName);
        }

        DTreesImplForBoostOwn impl;
    };
}