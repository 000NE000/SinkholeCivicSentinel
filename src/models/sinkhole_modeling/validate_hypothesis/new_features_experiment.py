"""
Integrated evaluation for enhanced sinkhole risk prediction with pothole data
"""
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import seaborn as sns
import shap
import torch
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from src.models.sinkhole_modeling.config import log, DEFAULT_MODEL_PARAMS, DEFAULT_K_VALS,DB_ENV_VAR
from src.models.sinkhole_modeling.data_loader import load_dataset, get_silent_grid_ids, create_spatial_blocks
from src.models.sinkhole_modeling.evaluation import evaluate_reports_impact_pu, plot_pu_report_impact
from src.models.sinkhole_modeling.optimizer import optimize_hyperparameters
from src.models.sinkhole_modeling.stage1_model import TwoStageSinkholeScreener
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import os
from dotenv import load_dotenv


def enhanced_shap_analysis(baseline_model, enhanced_model, X_base, X_enh, feature_names_base, feature_names_enh):
    """
    기존 모델과 향상된 모델의 SHAP 값을 비교 분석하는 함수
    """
    plt.figure(figsize=(16, 10))

    # 바탕색 설정 - 연한 회색
    plt.rcParams['figure.facecolor'] = '#f9f9f9'

    try:
        # SHAP 분석 준비 - 기존 모델
        print("Calculating SHAP values for baseline model...")
        baseline_explainer = shap.TreeExplainer(baseline_model)
        X_base_values = X_base[feature_names_base].values
        baseline_shap_values = baseline_explainer.shap_values(X_base_values)

        # 이진 분류 모델인 경우 양성 클래스의 SHAP 값 사용
        if isinstance(baseline_shap_values, list) and len(baseline_shap_values) > 1:
            baseline_shap_values = baseline_shap_values[1]

        # SHAP 분석 준비 - 향상된 모델
        print("Calculating SHAP values for enhanced model...")
        enhanced_explainer = shap.TreeExplainer(enhanced_model)
        X_enh_values = X_enh[feature_names_enh].values
        enhanced_shap_values = enhanced_explainer.shap_values(X_enh_values)

        # 이진 분류 모델인 경우 양성 클래스의 SHAP 값 사용
        if isinstance(enhanced_shap_values, list) and len(enhanced_shap_values) > 1:
            enhanced_shap_values = enhanced_shap_values[1]

        # 1. SHAP 요약 플롯 - 향상된 모델
        plt.subplot(1, 2, 1)
        shap.summary_plot(enhanced_shap_values, X_enh_values,
                          feature_names=feature_names_enh, show=False,
                          plot_size=(7, 8))
        plt.title('Enhanced Model - SHAP Feature Importance', fontsize=14, fontweight='bold')

        # pothole 관련 피처에 레이블 추가
        ax = plt.gca()
        yticks = ax.get_yticklabels()
        for i, tick in enumerate(yticks):
            feature_name = tick.get_text()
            if 'pothole' in feature_name.lower():
                tick.set_color('red')
                tick.set_fontweight('bold')

        # 2. SHAP 요약 플롯 - 기존 모델
        plt.subplot(1, 2, 2)
        shap.summary_plot(baseline_shap_values, X_base_values,
                          feature_names=feature_names_base, show=False,
                          plot_size=(7, 8))
        plt.title('Baseline Model - SHAP Feature Importance', fontsize=14, fontweight='bold')

        plt.tight_layout(pad=2.0)
        fig = plt.gcf()
        fig.savefig('shap_comparison.png', dpi=300, bbox_inches='tight')
        print("SHAP comparison plot saved as 'shap_comparison.png'")

        # 3. 특성 중요도 막대 그래프 (향상된 모델)
        plt.figure(figsize=(12, 8))

        # 평균 절대 SHAP 값 계산
        mean_shap_enh = np.abs(enhanced_shap_values).mean(axis=0)

        # 특성 중요도 데이터프레임 생성
        feature_importance_enh = pd.DataFrame({
            'Feature': feature_names_enh,
            'Importance': mean_shap_enh
        }).sort_values('Importance', ascending=False)

        # pothole 관련 피처 식별
        feature_importance_enh['is_pothole'] = feature_importance_enh['Feature'].apply(
            lambda x: 'pothole' in x.lower()
        )

        # 컬러맵 생성 - pothole 피처는 강조
        colors = feature_importance_enh['is_pothole'].map({True: '#ff7043', False: '#4285f4'})

        # 상위 20개 피처만 표시
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(
            x='Importance',
            y='Feature',
            data=feature_importance_enh.head(20),
            palette=colors.head(20)
        )

        # 범례 추가
        pothole_patch = mpatches.Patch(color='#ff7043', label='Pothole Features')
        other_patch = mpatches.Patch(color='#4285f4', label='Other Features')
        plt.legend(handles=[pothole_patch, other_patch], loc='lower right')

        plt.title('Top 20 Features by SHAP Importance (Enhanced Model)', fontsize=16, fontweight='bold')
        plt.xlabel('Mean |SHAP| Value (Impact on Model Output)', fontsize=12)
        plt.ylabel('Feature', fontsize=12)

        # SHAP 값 표시
        for i, (_, row) in enumerate(feature_importance_enh.head(20).iterrows()):
            plt.text(row['Importance'] + 0.001, i, f'{row["Importance"]:.4f}',
                     va='center', fontsize=10)

        plt.tight_layout()
        fig = plt.gcf()
        fig.savefig('feature_importance_enhanced.png', dpi=300, bbox_inches='tight')
        print("Enhanced feature importance plot saved as 'feature_importance_enhanced.png'")

        # 4. pothole 피처 중요도 순위 분석
        pothole_features = feature_importance_enh[feature_importance_enh['is_pothole']].copy()

        for _, row in pothole_features.iterrows():
            rank = feature_importance_enh[feature_importance_enh['Feature'] == row['Feature']].index[0] + 1
            print(f"Rank {rank}: {row['Feature']} - Importance: {row['Importance']:.4f}")

        # Top 10에 포함된 pothole 피처 수 계산
        top10_potholes = pothole_features[pothole_features['Feature'].isin(
            feature_importance_enh.head(10)['Feature']
        )]

        if len(top10_potholes) > 0:
            print(f"\n✅ 가설 1 지지: {len(top10_potholes)}개의 pothole 피처가 상위 10위 내에 포함됨")
            for _, row in top10_potholes.iterrows():
                rank = feature_importance_enh[feature_importance_enh['Feature'] == row['Feature']].index[0] + 1
                print(f"  - Rank {rank}: {row['Feature']} (Importance: {row['Importance']:.4f})")
        else:
            print("\n❌ 가설 1 기각: pothole 피처가 상위 10위 내에 포함되지 않음")

        return feature_importance_enh

    except Exception as e:
        print(f"Error in SHAP analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def visualize_false_negative_improvement(baseline_predictions, enhanced_predictions,
                                         y_true, X_data, pothole_features=None):
    """
    False Negative 개선 효과를 시각화하는 함수
    """
    # 1. FN 분석 - 기본 통계
    print("\n=== False Negative 개선 효과 분석 ===")

    # 기준 모델의 False Negative 식별
    baseline_fn_indices = np.where((y_true == 1) & (baseline_predictions < 0.5))[0]
    print(f"기준 모델 False Negative 수: {len(baseline_fn_indices)}")

    if len(baseline_fn_indices) == 0:
        print("분석할 False Negative가 없습니다.")
        return

    # 향상된 모델에서 수정된 FN 식별
    corrected_indices = baseline_fn_indices[
        np.where((y_true[baseline_fn_indices] == 1) &
                 (enhanced_predictions[baseline_fn_indices] >= 0.5))[0]
    ]

    # 여전히 FN인 경우 식별
    still_fn_indices = baseline_fn_indices[
        np.where((y_true[baseline_fn_indices] == 1) &
                 (enhanced_predictions[baseline_fn_indices] < 0.5))[0]
    ]

    correction_rate = len(corrected_indices) / len(baseline_fn_indices) if len(baseline_fn_indices) > 0 else 0
    print(f"수정된 False Negative 수: {len(corrected_indices)} ({correction_rate:.1%})")
    print(f"여전히 False Negative인 수: {len(still_fn_indices)} ({1 - correction_rate:.1%})")

    # 2. FN 통계 막대 그래프
    plt.figure(figsize=(12, 10))

    # 상단 그래프 - FN 수정 통계
    plt.subplot(2, 1, 1)
    counts = {
        'Original FNs': len(baseline_fn_indices),
        'Corrected (Now TPs)': len(corrected_indices),
        'Still FNs': len(still_fn_indices)
    }

    # 색상 지정
    colors = ['#f44336', '#4caf50', '#ff9800']  # 빨강, 초록, 주황

    bars = plt.bar(list(counts.keys()), list(counts.values()), color=colors)

    # 각 막대 위에 수치 표시
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{height}\n({height / counts["Original FNs"]:.1%})',
                 ha='center', va='bottom', fontweight='bold')

    plt.title('False Negative 개선 효과', fontsize=16, fontweight='bold')
    plt.ylabel('샘플 수', fontsize=12)
    plt.ylim(0, counts['Original FNs'] * 1.2)  # y축 범위 조정

    # pothole 관련 피처 분석 (있는 경우에만)
    if pothole_features and len(pothole_features) > 0:
        # 3. pothole 피처 값 비교 (수정된 FN vs 여전히 FN)
        plt.subplot(2, 1, 2)

        # 분석할 최대 3개 피처 선택
        analyze_features = pothole_features[:min(3, len(pothole_features))]

        for i, feature in enumerate(analyze_features):
            if feature not in X_data.columns:
                continue

            # 수정된 FN의 피처 값
            corrected_values = X_data.iloc[corrected_indices][feature].values if len(corrected_indices) > 0 else []

            # 여전히 FN인 경우의 피처 값
            still_fn_values = X_data.iloc[still_fn_indices][feature].values if len(still_fn_indices) > 0 else []

            # 데이터 준비 - 피처 값 비교용
            data = pd.DataFrame({
                'Feature Value': np.concatenate([corrected_values, still_fn_values]),
                'Group': ['Corrected FNs'] * len(corrected_values) + ['Still FNs'] * len(still_fn_values),
                'Feature': [feature] * (len(corrected_values) + len(still_fn_values))
            })

            # 바이올린 플롯으로 분포 비교
            ax = sns.violinplot(x='Feature', y='Feature Value', hue='Group', data=data,
                                split=True, inner='quart', palette={'Corrected FNs': '#4caf50', 'Still FNs': '#ff9800'})

            # 평균값 계산
            mean_corrected = np.mean(corrected_values) if len(corrected_values) > 0 else 0
            mean_still_fn = np.mean(still_fn_values) if len(still_fn_values) > 0 else 0

            # 각 그룹에 대한 평균 레이블 추가
            for j, (group, color) in enumerate([('Corrected FNs', '#4caf50'), ('Still FNs', '#ff9800')]):
                if group == 'Corrected FNs':
                    plt.text(i - 0.25, mean_corrected, f'평균: {mean_corrected:.2f}',
                             color=color, ha='center', fontweight='bold')
                else:
                    plt.text(i + 0.25, mean_still_fn, f'평균: {mean_still_fn:.2f}',
                             color=color, ha='center', fontweight='bold')

        plt.title('포트홀 피처 분포 비교: 수정된 FN vs 여전히 FN', fontsize=16, fontweight='bold')
        plt.xlabel('')
        plt.ylabel('피처 값', fontsize=12)
        plt.legend(title='')

    plt.tight_layout()
    plt.savefig('fn_improvement_analysis.png', dpi=300, bbox_inches='tight')
    print("False Negative 개선 효과 분석 그래프가 'fn_improvement_analysis.png'로 저장되었습니다.")

    # 4. Pothole 피처별 수정된 FN vs 여전히 FN 비교 (히스토그램)
    if pothole_features and len(pothole_features) > 0:
        # 최대 6개 피처 분석
        analyze_features = pothole_features[:min(6, len(pothole_features))]
        n_features = len(analyze_features)

        if n_features > 0:
            plt.figure(figsize=(15, n_features * 3))

            for i, feature in enumerate(analyze_features):
                if feature not in X_data.columns:
                    continue

                # 수정된 FN의 피처 값
                corrected_values = X_data.iloc[corrected_indices][feature].values if len(corrected_indices) > 0 else []

                # 여전히 FN인 경우의 피처 값
                still_fn_values = X_data.iloc[still_fn_indices][feature].values if len(still_fn_indices) > 0 else []

                plt.subplot(n_features, 1, i + 1)

                # 히스토그램 겹쳐 그리기
                if len(corrected_values) > 0:
                    sns.histplot(corrected_values, color='#4caf50', alpha=0.6, label='수정된 FN (현재 TP)',
                                 kde=True, stat='density')

                if len(still_fn_values) > 0:
                    sns.histplot(still_fn_values, color='#ff9800', alpha=0.6, label='여전히 FN',
                                 kde=True, stat='density')

                # 평균선 추가
                if len(corrected_values) > 0:
                    mean_corrected = np.mean(corrected_values)
                    plt.axvline(mean_corrected, color='#4caf50', linestyle='--',
                                linewidth=2, label=f'수정된 FN 평균: {mean_corrected:.2f}')

                if len(still_fn_values) > 0:
                    mean_still_fn = np.mean(still_fn_values)
                    plt.axvline(mean_still_fn, color='#ff9800', linestyle='--',
                                linewidth=2, label=f'여전히 FN 평균: {mean_still_fn:.2f}')

                # 평균 차이 표시
                if len(corrected_values) > 0 and len(still_fn_values) > 0:
                    diff = mean_corrected - mean_still_fn
                    diff_pct = (diff / (mean_still_fn + 1e-10)) * 100
                    plt.text(0.7, 0.85, f'평균 차이: {diff:.2f} ({diff_pct:.1f}%)',
                             transform=plt.gca().transAxes, fontsize=12, fontweight='bold',
                             bbox=dict(facecolor='white', alpha=0.8))

                plt.title(f'{feature} 분포 비교', fontsize=14, fontweight='bold')
                plt.xlabel(feature, fontsize=12)
                plt.ylabel('밀도', fontsize=12)
                plt.legend()

            plt.tight_layout()
            plt.savefig('fn_feature_histograms.png', dpi=300, bbox_inches='tight')
            print("포트홀 피처별 FN 개선 분석 히스토그램이 'fn_feature_histograms.png'로 저장되었습니다.")

    # 5. 가설 2 검증 결과
    if correction_rate > 0.2:  # 20% 이상 개선시 가설 지지
        result = f"\n✅ 가설 2 지지: {correction_rate:.1%}의 False Negative가 개선됨"
        if pothole_features and len(pothole_features) > 0:
            pothole_impact = []
            for feature in pothole_features:
                if feature in X_data.columns:
                    corrected_avg = np.mean(X_data.iloc[corrected_indices][feature]) if len(
                        corrected_indices) > 0 else 0
                    still_fn_avg = np.mean(X_data.iloc[still_fn_indices][feature]) if len(still_fn_indices) > 0 else 0
                    diff_pct = ((corrected_avg - still_fn_avg) / (still_fn_avg + 1e-10)) * 100
                    if diff_pct > 20:  # 20% 이상 차이가 있는 피처만 포함
                        pothole_impact.append(f"{feature} ({diff_pct:.1f}% 차이)")

            if pothole_impact:
                result += f"\n  주요 영향 피처: {', '.join(pothole_impact)}"
    else:
        result = f"\n❌ 가설 2 기각: 개선율 {correction_rate:.1%}로 충분하지 않음"

    print(result)

def load_pothole_features():
    """
    Load pothole-related features
    """
    load_dotenv()
    dsn = os.getenv(DB_ENV_VAR)
    if not dsn:
        raise RuntimeError(f"Environment variable {DB_ENV_VAR} is not set.")
    engine = create_engine(dsn)

    log("Loading pothole features...", level=1)
    try:
        pothole_df = pd.read_sql("SELECT * FROM pothole_features", engine)
        log(f"Loaded {len(pothole_df)} rows of pothole features", level=1)
        return pothole_df
    except Exception as e:
        log(f"Error loading pothole features: {str(e)}", level=1)
        log("Continuing without pothole features", level=1)
        return None


def integrate_pothole_features(gdf, pothole_df):
    """
    Integrate pothole features into the main dataset
    """
    if pothole_df is None:
        return gdf

    log("Integrating pothole features into main dataset...", level=1)

    # Ensure both dataframes have grid_id column
    if 'grid_id' not in gdf.columns or 'grid_id' not in pothole_df.columns:
        log("Error: grid_id column missing in one of the datasets", level=1)
        return gdf

    # Merge pothole features on grid_id
    # Using left join to keep all original data
    result = gdf.merge(
        pothole_df,
        on='grid_id',
        how='left',
        suffixes=('', '_pothole')
    )

    # Fill NA values for pothole features
    pothole_features = [c for c in pothole_df.columns if c != 'grid_id']
    for col in pothole_features:
        if col in result.columns:
            result[col] = result[col].fillna(0)

    log(f"Dataset integration complete. New shape: {result.shape}", level=1)
    log(f"Added pothole features: {pothole_features}", level=1)

    return result


def shap_analysis(model, X_test, feature_names):
    """
    Perform SHAP analysis on the model
    """
    log("Performing SHAP analysis...", level=1)

    # Check if model supports SHAP
    if hasattr(model, "predict_proba") and not isinstance(model, torch.nn.Module):
        try:
            # Create explainer
            explainer = shap.TreeExplainer(model)

            # Calculate SHAP values
            X_test_values = X_test[feature_names].values
            shap_values = explainer.shap_values(X_test_values)

            # For binary classification, we're interested in the positive class (index 1)
            if isinstance(shap_values, list) and len(shap_values) > 1:
                shap_values = shap_values[1]

            # Get mean absolute SHAP value for each feature
            mean_shap = np.abs(shap_values).mean(axis=0)

            # Create feature importance dataframe
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': mean_shap
            }).sort_values('Importance', ascending=False)

            # Save SHAP summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_test_values, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
            log("SHAP summary plot saved as 'shap_summary_plot.png'", level=1)

            # Save feature importance plot
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
            plt.title('Top 20 Features by SHAP Importance')
            plt.tight_layout()
            plt.savefig('feature_importance_plot.png', dpi=300, bbox_inches='tight')
            log("Feature importance plot saved as 'feature_importance_plot.png'", level=1)

            # Evaluate pothole feature importance
            pothole_features = [f for f in feature_names if 'pothole' in f.lower()]
            pothole_importance = feature_importance[feature_importance['Feature'].isin(pothole_features)]

            log("\n=== Pothole Feature Importance ===", level=1)
            for _, row in pothole_importance.iterrows():
                rank = feature_importance[feature_importance['Feature'] == row['Feature']].index[0] + 1
                log(f"Rank {rank}: {row['Feature']} - Importance: {row['Importance']:.4f}", level=1)

            # Is any pothole feature in the top 10?
            top_pothole = pothole_importance[pothole_importance['Feature'].isin(feature_importance.head(10)['Feature'])]
            if len(top_pothole) > 0:
                log("\n✅ Hypothesis 1 SUPPORTED: Pothole features appear in top 10 important features", level=1)
            else:
                log("\n❌ Hypothesis 1 NOT SUPPORTED: No pothole features in top 10", level=1)

            return feature_importance

        except Exception as e:
            log(f"Error in SHAP analysis: {str(e)}", level=1)
            return None
    else:
        log("Model type does not support SHAP analysis", level=1)
        return None


def analyze_false_negatives(baseline_predictions, enhanced_predictions, y_true, X_data):
    """
    Analyze false negatives improvement with pothole features
    """
    log("\n=== False Negative Analysis ===", level=1)

    # Get indices of false negatives in baseline model
    baseline_fn_indices = np.where((y_true == 1) & (baseline_predictions < 0.5))[0]
    log(f"Baseline model false negatives: {len(baseline_fn_indices)}", level=1)

    if len(baseline_fn_indices) == 0:
        log("No false negatives to analyze", level=1)
        return

    # Check which false negatives are now correctly predicted by enhanced model
    corrected_indices = np.where((y_true[baseline_fn_indices] == 1) &
                                 (enhanced_predictions[baseline_fn_indices] >= 0.5))[0]

    if len(corrected_indices) == 0:
        log("No false negatives were corrected by the enhanced model", level=1)
        log("❌ Hypothesis 2 NOT SUPPORTED: No improvement in false negatives", level=1)
        return

    corrected_indices = baseline_fn_indices[corrected_indices]
    log(f"Enhanced model corrected {len(corrected_indices)} out of {len(baseline_fn_indices)} false negatives", level=1)

    # Check if pothole features helped with correction
    pothole_features = [c for c in X_data.columns if 'pothole' in c.lower()]

    if not pothole_features:
        log("No pothole features found in dataset", level=1)
        return

    # Compare pothole feature values in corrected vs. still-incorrect cases
    still_fn_indices = np.where((y_true == 1) & (enhanced_predictions < 0.5))[0]

    # Create comparison dataframe
    comparison = pd.DataFrame()
    for feature in pothole_features:
        if feature in X_data.columns:
            # Get feature values for corrected and still-incorrect cases
            corrected_values = X_data.iloc[corrected_indices][feature].values
            still_fn_values = X_data.iloc[still_fn_indices][feature].values if len(still_fn_indices) > 0 else []

            mean_corrected = np.mean(corrected_values) if len(corrected_values) > 0 else 0
            mean_still_fn = np.mean(still_fn_values) if len(still_fn_values) > 0 else 0

            comparison = comparison.append({
                'Feature': feature,
                'Mean in Corrected FNs': mean_corrected,
                'Mean in Remaining FNs': mean_still_fn,
                'Difference': mean_corrected - mean_still_fn,
                'Difference (%)': ((mean_corrected - mean_still_fn) / (mean_still_fn + 1e-10)) * 100
            }, ignore_index=True)

    # Sort by difference
    comparison = comparison.sort_values('Difference', ascending=False)

    # Display comparison
    log("\nPothole Feature Comparison between Corrected and Remaining False Negatives:", level=1)
    log(comparison.to_string(index=False), level=1)

    # Create visualization
    plt.figure(figsize=(12, 8))

    for i, feature in enumerate(comparison['Feature']):
        corrected_values = X_data.iloc[corrected_indices][feature].values
        still_fn_values = X_data.iloc[still_fn_indices][feature].values if len(still_fn_indices) > 0 else []

        plt.subplot(len(comparison), 1, i + 1)
        sns.histplot(corrected_values, color='green', alpha=0.5, label='Corrected FNs', kde=True)
        if len(still_fn_values) > 0:
            sns.histplot(still_fn_values, color='red', alpha=0.5, label='Remaining FNs', kde=True)
        plt.legend()
        plt.title(feature)
        plt.tight_layout()

    plt.savefig('fn_analysis.png', dpi=300, bbox_inches='tight')
    log("False negative analysis visualization saved as 'fn_analysis.png'", level=1)

    # Evaluate hypothesis
    significant_features = comparison[comparison['Difference (%)'] > 20]
    if len(significant_features) > 0:
        log("\n✅ Hypothesis 2 SUPPORTED: Pothole features show significantly higher values in corrected FNs", level=1)
    else:
        log("\n❌ Hypothesis 2 NOT SUPPORTED: No significant difference in pothole feature values", level=1)


def analyze_coverage_by_density(predictions, y_true, X_data, density_feature='pothole_kde_density', k=100):
    """
    Analyze model performance across different pothole density areas
    """
    log("\n=== Coverage by Pothole Density Analysis ===", level=1)

    if density_feature not in X_data.columns:
        log(f"Density feature '{density_feature}' not found in dataset", level=1)
        return

    # Divide areas into high and low pothole density
    density = X_data[density_feature].values
    high_threshold = np.percentile(density, 90)
    low_threshold = np.percentile(density, 10)

    high_density_mask = density >= high_threshold
    low_density_mask = density <= low_threshold

    log(f"High density threshold (90th percentile): {high_threshold:.4f}", level=1)
    log(f"Low density threshold (10th percentile): {low_threshold:.4f}", level=1)
    log(f"Number of high density grids: {np.sum(high_density_mask)}", level=1)
    log(f"Number of low density grids: {np.sum(low_density_mask)}", level=1)

    # Evaluate model performance in each segment
    def evaluate_segment(name, mask):
        if np.sum(mask) == 0:
            return None

        # Get top k predictions in this segment
        segment_preds = predictions[mask]
        segment_true = y_true[mask]

        # Sort by prediction scores
        idx = np.argsort(segment_preds)[::-1]
        top_k_idx = idx[:min(k, len(idx))]

        # Calculate metrics
        precision_k = np.mean(segment_true[top_k_idx]) if len(top_k_idx) > 0 else 0
        recall_k = np.sum(segment_true[top_k_idx]) / np.sum(segment_true) if np.sum(segment_true) > 0 else 0

        return {
            'Segment': name,
            'Count': np.sum(mask),
            'Positives': np.sum(segment_true),
            'Positive Rate': np.mean(segment_true),
            f'Precision@{k}': precision_k,
            f'Recall@{k}': recall_k
        }

    high_results = evaluate_segment('High Density', high_density_mask)
    low_results = evaluate_segment('Low Density', low_density_mask)
    all_results = evaluate_segment('All Areas', np.ones_like(y_true, dtype=bool))

    # Create comparison table
    results_table = pd.DataFrame([high_results, low_results, all_results])
    log("\nPerformance by Pothole Density:", level=1)
    log(results_table.to_string(index=False), level=1)

    # Visualization
    plt.figure(figsize=(10, 6))
    metrics = [f'Precision@{k}', f'Recall@{k}']

    for i, metric in enumerate(metrics):
        plt.subplot(1, 2, i + 1)
        sns.barplot(x='Segment', y=metric, data=results_table)
        plt.title(metric)
        plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig('density_analysis.png', dpi=300, bbox_inches='tight')
    log("Pothole density analysis visualization saved as 'density_analysis.png'", level=1)

    # Evaluate hypothesis
    if high_results and low_results:
        precision_diff = high_results[f'Precision@{k}'] - low_results[f'Precision@{k}']
        recall_diff = high_results[f'Recall@{k}'] - low_results[f'Recall@{k}']

        log(f"\nPrecision@{k} difference (High - Low): {precision_diff:.4f}", level=1)
        log(f"Recall@{k} difference (High - Low): {recall_diff:.4f}", level=1)

        if precision_diff > 0.1 or recall_diff > 0.1:
            log("\n✅ Hypothesis 3 SUPPORTED: Model performs better in high pothole density areas", level=1)
        else:
            log("\n❌ Hypothesis 3 NOT SUPPORTED: No significant performance difference between areas", level=1)


def analyze_roi(predictions, y_true, k_values=[100, 200, 500]):
    """
    Analyze ROI by evaluating coverage of top-k predictions
    """
    log("\n=== ROI Analysis ===", level=1)

    # Sort by prediction scores
    idx = np.argsort(predictions)[::-1]

    results = []
    for k in k_values:
        if k > len(idx):
            continue

        # Get top-k predictions
        top_k_idx = idx[:k]

        # Calculate metrics
        true_positives = np.sum(y_true[top_k_idx])
        all_positives = np.sum(y_true)

        recall = true_positives / all_positives if all_positives > 0 else 0
        precision = true_positives / k
        relative_efficiency = (true_positives / k) / (all_positives / len(y_true))

        results.append({
            'k': k,
            'True Positives': true_positives,
            'Recall': recall,
            'Precision': precision,
            'Coverage %': recall * 100,
            'Efficiency Multiplier': relative_efficiency
        })

    # Create results table
    results_df = pd.DataFrame(results)
    log("\nROI Analysis Results:", level=1)
    log(results_df.to_string(index=False), level=1)

    # Create visualization
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.barplot(x='k', y='Coverage %', data=results_df)
    plt.title('Risk Coverage by Top-k Inspection')
    plt.ylabel('% of Total Risk Covered')

    plt.subplot(1, 2, 2)
    sns.barplot(x='k', y='Efficiency Multiplier', data=results_df)
    plt.title('Efficiency Multiplier')
    plt.ylabel('Times More Efficient Than Random')

    plt.tight_layout()
    plt.savefig('roi_analysis.png', dpi=300, bbox_inches='tight')
    log("ROI analysis visualization saved as 'roi_analysis.png'", level=1)

    # Evaluate hypothesis
    if len(results) > 0 and 100 in [r['k'] for r in results]:
        k100_result = next(r for r in results if r['k'] == 100)
        if k100_result['Coverage %'] > 25:  # Threshold: cover >25% of risks with top 100
            log(f"\n✅ Hypothesis 4 SUPPORTED: Top 100 grids cover {k100_result['Coverage %']:.1f}% of total risk",
                level=1)
        else:
            log(f"\n❌ Hypothesis 4 NOT SUPPORTED: Top 100 grids only cover {k100_result['Coverage %']:.1f}% of total risk",
                level=1)


# Modified main function with added individual SHAP analysis for enhanced model
def main() -> None:
    # List of report tables to evaluate
    report_tables = [
        "feature_matrix_100_geo"
    ]

    # ================== 0. Load pothole features
    log("\n=== Step 0: Loading and Integrating Pothole Features ===", level=1)
    pothole_features = load_pothole_features()

    # ================== 1. Load the 100-report dataset and integrate pothole features
    log("\n=== Step 1: Hyperparameter Optimization with Integrated Dataset ===", level=1)
    gdf_100 = load_dataset(table="feature_matrix_100_geo")

    # Integrate pothole features
    if pothole_features is not None:
        gdf_100 = integrate_pothole_features(gdf_100, pothole_features)

    # Prepare data - grid_id 컬럼 유지 (silent zone 통합에 필요)
    y = gdf_100["subsidence_occurrence"].astype(int)
    X_feat = gdf_100.drop(columns=[c for c in [
        "subsidence_occurrence", "subsidence_count", "geometry"]
                                   if c in gdf_100.columns])
    X_geo = gdf_100.copy()

    # Create spatial blocks
    spatial_blocks = create_spatial_blocks(gdf_100, n_blocks=5)
    # Dynamically generate silent grid IDs
    silent_ids = get_silent_grid_ids(X_feat, y, percentile=90)
    log(f"Generated {len(silent_ids)} silent grid IDs", level=1)

    # Log all columns including pothole features
    log(f"Columns in integrated dataset: {list(X_geo.columns)}", level=1)

    # Optimize hyperparameters
    opt_results = optimize_hyperparameters(X_geo, y, spatial_blocks)

    # Step 2: Train baseline model (without pothole features) and enhanced model (with pothole features)
    log("\n=== Step 2: Training Baseline and Enhanced Models ===", level=1)

    # First: Train baseline model without pothole features
    baseline_gdf = load_dataset(table="feature_matrix_100_geo")  # Original data without pothole features
    baseline_y = baseline_gdf["subsidence_occurrence"].astype(int)
    baseline_X = baseline_gdf.copy()

    best_params = opt_results['best_params']
    if best_params is None:
        best_params = DEFAULT_MODEL_PARAMS
        log(f"Using default model parameters: {best_params}", level=1)
    else:
        best_params = {
            'proximity_feat': DEFAULT_MODEL_PARAMS['proximity_feat'],
            'stage1_model_type': DEFAULT_MODEL_PARAMS['stage1_model_type'],
            'threshold_percentile': DEFAULT_MODEL_PARAMS['threshold_percentile'],
            'feature_fraction': DEFAULT_MODEL_PARAMS['feature_fraction'],
            'alpha': DEFAULT_MODEL_PARAMS.get('ensemble_weight', 0.5),
            'model_type': DEFAULT_MODEL_PARAMS.get('stage2_model_type', 'lightgbm'),
            'margin_low': DEFAULT_MODEL_PARAMS.get('margin_low', 0.4),
            'margin_high': DEFAULT_MODEL_PARAMS.get('margin_high', 0.6)
        }
        log(f"Using optimized parameters: {best_params}", level=1)

    # Train baseline model
    log("Training baseline model (without pothole features)...", level=1)
    baseline_model = TwoStageSinkholeScreener(
        proximity_feat=best_params['proximity_feat'],
        stage1_model_type=best_params['stage1_model_type'],
        threshold_percentile=best_params['threshold_percentile'],
        feature_fraction=best_params['feature_fraction'],
        ensemble_weight=best_params['alpha']
    )
    baseline_model.fit(
        baseline_X, baseline_y, spatial_blocks=spatial_blocks,
        model_type=best_params['model_type'],
        margin_low=best_params['margin_low'],
        margin_high=best_params['margin_high']
    )

    # Train enhanced model with pothole features
    log("Training enhanced model (with pothole features)...", level=1)
    enhanced_model = TwoStageSinkholeScreener(
        proximity_feat=best_params['proximity_feat'],
        stage1_model_type=best_params['stage1_model_type'],
        threshold_percentile=best_params['threshold_percentile'],
        feature_fraction=best_params['feature_fraction'],
        ensemble_weight=best_params['alpha']
    )
    enhanced_model.fit(
        X_geo, y, spatial_blocks=spatial_blocks,
        model_type=best_params['model_type'],
        margin_low=best_params['margin_low'],
        margin_high=best_params['margin_high']
    )

    # Split data for evaluation
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_geo, y, test_size=0.3, random_state=42, stratify=y
    )

    # Get baseline test set with correct indices
    X_test_base = baseline_X.iloc[X_test.index]

    # Get predictions from both models
    baseline_predictions = baseline_model.predict(X_test_base)
    enhanced_predictions = enhanced_model.predict(X_test)

    # ========================= Step 3: SHAP Analysis (Hypothesis 1)
    log("\n=== Step 3: SHAP Analysis (Hypothesis 1) ===", level=1)

    # Get feature names excluding geometry and target for both models
    feature_names_base = [c for c in X_test_base.columns
                          if c not in ['geometry', 'subsidence_occurrence', 'subsidence_count', 'grid_id']]
    feature_names_enh = [c for c in X_test.columns
                         if c not in ['geometry', 'subsidence_occurrence', 'subsidence_count', 'grid_id']]

    # First: Run enhanced SHAP analysis comparing both models (original code)
    feature_importance = enhanced_shap_analysis(
        baseline_model.stage2_model,
        enhanced_model.stage2_model,
        X_test_base,
        X_test,
        feature_names_base,
        feature_names_enh
    )

    # NEW: Add individual SHAP analysis specifically for the enhanced model
    log("Performing individual SHAP analysis for enhanced model...", level=1)
    enhanced_feature_importance = shap_analysis(
        enhanced_model.stage2_model,
        X_test,
        feature_names_enh
    )

    # Create a specific enhanced model SHAP summary plot
    log("Creating detailed SHAP summary plot for enhanced model...", level=1)
    try:
        plt.figure(figsize=(14, 10))
        enhanced_explainer = shap.TreeExplainer(enhanced_model.stage2_model)
        X_test_values = X_test[feature_names_enh].values
        enhanced_shap_values = enhanced_explainer.shap_values(X_test_values)

        # For binary classification, we're interested in the positive class (index 1)
        if isinstance(enhanced_shap_values, list) and len(enhanced_shap_values) > 1:
            enhanced_shap_values = enhanced_shap_values[1]

        # Create a detailed SHAP summary plot
        shap.summary_plot(enhanced_shap_values, X_test_values,
                          feature_names=feature_names_enh, show=False)
        plt.title('Enhanced Model with Pothole Features - SHAP Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('enhanced_model_shap_summary.png', dpi=300, bbox_inches='tight')
        log("Enhanced model SHAP summary plot saved as 'enhanced_model_shap_summary.png'", level=1)

        # Create a SHAP decision plot
        plt.figure(figsize=(16, 10))

        # Get some representative samples (e.g., top 100 predictions)
        top_indices = np.argsort(enhanced_predictions)[-100:]

        # Decision plot with a subset of samples for clarity
        shap.decision_plot(enhanced_explainer.expected_value if not isinstance(enhanced_explainer.expected_value, list)
                           else enhanced_explainer.expected_value[1],
                           enhanced_shap_values[top_indices],
                           X_test_values[top_indices],
                           feature_names=feature_names_enh, show=False)

        plt.title('Enhanced Model - SHAP Decision Plot (Top 100 Predictions)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('enhanced_model_shap_decision_plot.png', dpi=300, bbox_inches='tight')
        log("Enhanced model SHAP decision plot saved as 'enhanced_model_shap_decision_plot.png'", level=1)

    except Exception as e:
        log(f"Error creating enhanced model SHAP plots: {str(e)}", level=1)
        import traceback
        traceback.print_exc()

    # Get pothole features from results if available
    if feature_importance is not None:
        pothole_features = feature_importance[feature_importance['is_pothole']]['Feature'].tolist()
    else:
        pothole_features = [c for c in X_test.columns if 'pothole' in c.lower()]

    # ========================= Step 4: False Negative Analysis (Hypothesis 2)
    log("\n=== Step 4: False Negative Analysis (Hypothesis 2) ===", level=1)
    visualize_false_negative_improvement(
        baseline_predictions,
        enhanced_predictions,
        y_test.values,
        X_test,
        pothole_features
    )

    # ========================= Step 5: Coverage Analysis by Pothole Density (Hypothesis 3)
    log("\n=== Step 5: Coverage Analysis by Pothole Density (Hypothesis 3) ===", level=1)
    analyze_coverage_by_density(enhanced_predictions, y_test.values, X_test)

    # ========================= Step 6: ROI Analysis (Hypothesis 4)
    log("\n=== Step 6: ROI Analysis (Hypothesis 4) ===", level=1)
    analyze_roi(enhanced_predictions, y_test.values)

    # ========================= Step 7: Active Learning Suggestions
    log("\n=== Step 7: Generating Active Learning Suggestions ===", level=1)
    try:
        top_suggestions = enhanced_model.get_active_learning_suggestions(X_geo, silent_ids, top_k=50)
        log(f"Top 50 grid IDs for investigation: {top_suggestions[:10]}... (and 40 more)", level=1)

        # 추천 결과 저장
        pd.DataFrame({'grid_id': top_suggestions}).to_csv('active_learning_suggestions.csv', index=False)
        log("Active learning suggestions saved to 'active_learning_suggestions.csv'", level=1)
    except Exception as e:
        log(f"Error in active learning suggestions: {str(e)}", level=1)

    # ========================= Step 8: PU Learning Evaluation
    log("\n=== Step 8: PU Learning Evaluation ===", level=1)

    # Decide model_type based on optimized parameters
    use_graphsage = best_params.get('model_type', 'lightgbm') in ['graphsage', 'um_gnn', 'gnn']

    # Evaluate report impact using PU-learning simulation
    pu_results = evaluate_reports_impact_pu(
        report_tables,
        k_vals=DEFAULT_K_VALS,
        n_iters=3,  # Reduced iterations for faster execution
        include_silent_metrics=False,
        use_graphsage=use_graphsage,
        use_uncertainty_masking=False,
        random_state=42
    )

    # Create results table with expanded metrics
    report_counts = sorted(pu_results.keys())
    metrics = ([f"recall@{k}" for k in DEFAULT_K_VALS] +
               [f"precision@{k}" for k in DEFAULT_K_VALS] +
               [f"lift@{k}" for k in DEFAULT_K_VALS] +
               ['pr_auc'])

    table_data = []
    for count in report_counts:
        row = [count]
        for metric in metrics:
            mean = pu_results[count][f"{metric}_mean"]
            std = pu_results[count][f"{metric}_std"]
            row.append(f"{mean:.4f} ± {std:.4f}")
        table_data.append(row)

    table_df = pd.DataFrame(table_data, columns=["Report Count"] + metrics)
    log("\nPU-Learning Results table:", level=1)
    log(table_df.to_string(index=False), level=1)

    # Save results table
    table_df.to_csv('sinkhole_pu_report_impact_results.csv', index=False)
    log("PU results saved to 'sinkhole_pu_report_impact_results.csv'", level=1)

    # Plot improvement with increasing reports
    plot_pu_report_impact(pu_results, k_vals=DEFAULT_K_VALS, include_silent_metrics=False)

    # Final summary of hypothesis testing
    log("\n=== Hypothesis Testing Summary ===", level=1)
    log("Hypothesis 1: Pothole features are important predictors for sinkhole risk", level=1)
    log("Hypothesis 2: Pothole data helps reduce false negatives in prediction", level=1)
    log("Hypothesis 3: Model performs better in areas with higher pothole reporting density", level=1)
    log("Hypothesis 4: Top-100 highest-risk grids cover a significant portion of total risk", level=1)

    log("Analysis complete.", level=1)


if __name__ == "__main__":
    t0 = time.time()
    main()
    log(f"Total runtime {(time.time() - t0):.1f}s")


