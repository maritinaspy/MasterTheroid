#!/usr/bin/env python3
# run_sex_prediction.py
#
# Sex prediction (female/male) using FEATURE VECTORS + PCA (as your pipeline does),
# while making sure gender is NOT included as an input feature (no leakage).
#
# Requirements:
# - Put this script in the same folder as preProcessing.py (the file you uploaded),
#   OR adjust PYTHONPATH so it can import it.

import argparse
import os
import numpy as np
import pandas as pd

import preProcessing as pp

#def remove_gender_column_if_present(X, feat_names):
  
    #if feat_names is None:
        #return X, feat_names

    #fn = np.array(feat_names, dtype=str)

    #gender_names = {"gender", "Gender", "sex", "Sex"}
    #gender_idx = np.where(np.isin(fn, list(gender_names)))[0]

    #if X.shape[1] != len(fn):
        #print(f"⚠️ WARNING: X has {X.shape[1]} columns but feat_names has {len(fn)} names.")
        #print("   Θα αφαιρέσω gender με βάση indices (safe), όχι boolean mask.")

    #if gender_idx.size > 0:
        #found = set(fn[gender_idx])
        #print(f"⚠️ Gender-like feature column(s) detected in X: {found}. Removing for sex prediction.")

        # remove columns by index
        #X = np.delete(X, gender_idx, axis=1)
        #fn = np.delete(fn, gender_idx)

    #return X, fn.tolist()

def main():
    parser = argparse.ArgumentParser(
        description="Sex prediction using feature vectors + PCA (consistent with your ML pipeline)."
    )

    # Preprocessing flags (match your pipeline knobs)
    parser.add_argument("-p", "--postProcessing", action="store_true", default=False)
    parser.add_argument("-norm", "--normalization", action="store_true", default=False)
    parser.add_argument("-ls", "--logScale", action="store_true", default=False)
    parser.add_argument("-stdf", "--stdevFiltering", action="store_true", default=False)
    parser.add_argument("-nfeat", "--numberOfFeaturesPerLevel", type=int, default=50)

    # PCA settings (your pipeline uses PCA as "feature selection")
    parser.add_argument("--pca_components", type=int, default=100)

    # Models to run
    parser.add_argument("-dect", "--decisionTree", action="store_true", default=False)
    parser.add_argument("-knn", "--kneighbors", action="store_true", default=False)
    parser.add_argument("-xgb", "--xgboost", action="store_true", default=False)
    parser.add_argument("-randf", "--randomforest", action="store_true", default=False)
    parser.add_argument("-nv", "--naivebayes", action="store_true", default=False)
    parser.add_argument("-strdum", "--stratifieddummyclf", action="store_true", default=False)
    parser.add_argument("-mfdum", "--mostfrequentdummyclf", action="store_true", default=False)
    parser.add_argument("-mlp", "--mlpClassifier", action="store_true", default=False)

    parser.add_argument("--saveResults", action="store_true", default=False)
    parser.add_argument("--outCSV", type=str, default="saved_results_sex_prediction.csv")

    parser.add_argument("--dtExplain", action="store_true", default=False)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--outDTPC", type=str, default="dt_top_pcs.csv")
    parser.add_argument("--outDTFeat", type=str, default="dt_top_original_features.csv")

    args = parser.parse_args()

    # Keep consistent with your project
    os.chdir("/datastore/maritina/MasterTheroid/sex_prediction")

    print("=== Sex prediction (feature vectors + PCA) ===")
    print("Loading feature vectors with your preprocessing pipeline...")

    # Load features + labels from your existing code
    mFeatures, vClass, sampleIDs, feat_names, tumor_stage, gender = pp.initializeFeatureMatrices(
        bPostProcessing=args.postProcessing,
        bstdevFiltering=args.stdevFiltering,
        nfeat=args.numberOfFeaturesPerLevel,
        bNormalize=args.normalization,
        bNormalizeLog2Scale=args.logScale,
    )

    print(f"Loaded mFeatures shape: {mFeatures.shape}")
    print(f"Loaded feat_names length: {len(feat_names)}")
    print(f"Loaded gender length: {len(gender)}")

    # κρατάμε μόνο omics features (drop last 3 cols: gender, case, stage)
    X = mFeatures[:, :-3]

    # ευθυγράμμιση ονομάτων features (safe αν υπάρχει mismatch)
    feat_names = feat_names[:X.shape[1]]

    # Build y from gender, filter unknown/NA
    y = np.array(gender, dtype=int)
    
    #SANITY CHECK
    print("USING SLICE DROP LAST 3 COLS ✅")
    print("mFeatures:", mFeatures.shape)
    print("X:", X.shape)
    print("feat_names:", len(feat_names))
    print("gender:", len(gender), "unique:", np.unique(np.array(gender, dtype=int)))

    print(f"After filtering NA gender -> X shape: {X.shape}, y shape: {y.shape}")
    uniq, cnt = np.unique(y, return_counts=True)
    print("Gender distribution (label -> count):", dict(zip(uniq.tolist(), cnt.tolist())))

    # PCA (your pipeline uses PCA before ML)
    print(f"Running PCA with n_components={args.pca_components} ...")
    X_pca, pca = pp.getPCA(X, args.pca_components)
    print(f"PCA output shape: {X_pca.shape}")

    if args.dtExplain:
        from sklearn.tree import DecisionTreeClassifier
        import pandas as pd
        import numpy as np

        # Fit ένα DT σε όλο το PCA-space (μόνο για εξήγηση)
        dt = DecisionTreeClassifier(random_state=0)
        dt.fit(X_pca, y)

        importances_pc = dt.feature_importances_  # μήκος = n_components
        pc_names = [f"PC{i+1}" for i in range(len(importances_pc))]

        # 1) Top PCs
        df_pc = pd.DataFrame({"PC": pc_names, "importance": importances_pc})
        df_pc = df_pc.sort_values("importance", ascending=False).head(args.topk)
        df_pc.to_csv(args.outDTPC, index=False)
        print(f"✅ DT top PCs saved to: {args.outDTPC}")

        # 2) Approx mapping πίσω στα αρχικά features μέσω PCA loadings
        # pca.components_.shape = (n_components, n_original_features)
        # score(feature_i) = sum_j importance(PC_j) * |loading_{j,i}|
        loadings_abs = np.abs(pca.components_)  # (k, p)
        feat_scores = importances_pc @ loadings_abs  # (p,)

        df_feat = pd.DataFrame({
            "feature": feat_names,
            "score": feat_scores
        }).sort_values("score", ascending=False).head(args.topk)

        df_feat.to_csv(args.outDTFeat, index=False)
        print(f"✅ DT top ORIGINAL features (via PCA mapping) saved to: {args.outDTFeat}")

    metricResults = []
    savedResults = {}

    # Run chosen models (each uses your existing CV/evaluation functions)
    if args.decisionTree:
        print("Running Decision Tree...")
        pp.classify(X_pca, y, metricResults, "DT_FeatureV_Sex", savedResults)

    if args.kneighbors:
        print("Running KNN...")
        pp.kneighbors(X_pca, y, metricResults, "kNN_FeatureV_Sex", savedResults)

    if args.xgboost:
        print("Running XGBoost...")
        pp.xgboost(X_pca, y, metricResults, "XGB_FeatureV_Sex", savedResults)

    if args.randomforest:
        print("Running Random Forest...")
        pp.RandomForest(X_pca, y, metricResults, "RF_FeatureV_Sex", savedResults)

    if args.naivebayes:
        print("Running Naive Bayes...")
        pp.NBayes(X_pca, y, metricResults, "NV_FeatureV_Sex", savedResults)

    if args.stratifieddummyclf:
        print("Running Stratified Dummy Classifier...")
        pp.stratifiedDummyClf(X_pca, y, metricResults, "StratDummy_FeatureV_Sex", savedResults)

    if args.mostfrequentdummyclf:
        print("Running Most Frequent Dummy Classifier...")
        pp.mostFrequentDummyClf(X_pca, y, metricResults, "MFDummy_FeatureV_Sex", savedResults)

    if args.mlpClassifier:
        print("Running MLP Classifier...")
        pp.mlpClassifier(X_pca, y, metricResults, "MLP_FeatureV_Sex", savedResults)

    if not any([
        args.decisionTree, args.kneighbors, args.xgboost, args.randomforest,
        args.naivebayes, args.stratifieddummyclf, args.mostfrequentdummyclf, args.mlpClassifier
    ]):
        print("⚠️ No model flags given. Example: -dect -randf -xgb")

    print("=== Done ===")

    if args.saveResults:
        if len(savedResults) == 0:
            print("⚠️ Δεν υπάρχουν savedResults για αποθήκευση (μήπως δεν έτρεξε κανένα model;).")
        else:
            df = pd.DataFrame.from_dict(savedResults, orient="index").reset_index()
            df.rename(columns={"index": "Model"}, inplace=True)
            df.to_csv(args.outCSV, index=False)
            print(f"✅ Results saved to: {args.outCSV}")


if __name__ == "__main__":
    main()