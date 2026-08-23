import streamlit as st
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.data_processor import DataProcessor
from src.numerical_methods.matrix_solver import MatrixSolver
from src.numerical_methods.root_finder import RootFinder
from src.numerical_methods.eigen_solver import EigenSolver
from src.numerical_methods.differentiation import NumericalDifferentiation
from src.numerical_methods.integration import NumericalIntegration
from src.numerical_methods.ode_solver import ODESolver
from src.numerical_methods.bvp_solver import BVPSolver
from src.regression.regression_engine import RegressionEngine
from src.interpolation.interpolation_engine import InterpolationEngine
from src.error_analysis.error_analyzer import ErrorAnalyzer
from src.visualization.plotter import Plotter

st.set_page_config(
    page_title="House Price Prediction - Numerical Methods",
    page_icon=" ",
    layout="wide",
)

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "house_prices.csv")


def generate_synthetic_data(n=100):
    np.random.seed(42)
    area = np.random.randint(800, 5000, n).astype(float)
    bedrooms = np.random.randint(1, 7, n).astype(float)
    bathrooms = np.random.randint(1, 5, n).astype(float)
    age = np.random.randint(0, 50, n).astype(float)
    parking = np.random.randint(0, 3, n).astype(float)
    location_score = np.random.randint(1, 11, n).astype(float)
    distance_center = np.random.uniform(0.5, 30.0, n)
    price = (
        5000 * area
        + 1500000 * bedrooms
        + 800000 * bathrooms
        - 100000 * age
        + 500000 * parking
        + 600000 * location_score
        - 200000 * distance_center
        + np.random.normal(0, 500000, n)
    )
    df = pd.DataFrame({
        "area": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
        "age": age, "parking": parking, "location_score": location_score,
        "distance_center": distance_center, "price": price,
    })
    return df


def init_session():
    defaults = {
        "page": "Home",
        "df": None,
        "X": None,
        "y": None,
        "feature_names": None,
        "data_processor": DataProcessor(),
        "regression_engine": None,
        "model_fitted": False,
        "coefficients": None,
        "y_pred": None,
        "y_test": None,
        "y_train": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


def load_data(uploaded_file=None, use_default=False):
    dp = st.session_state.data_processor
    if uploaded_file is not None:
        tmp_path = os.path.join("/tmp/opencode", uploaded_file.name)
        os.makedirs(os.path.dirname(tmp_path), exist_ok=True)
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.read())
        dp.load_csv(tmp_path)
    elif use_default and os.path.exists(DEFAULT_DATA_PATH):
        dp.load_csv(DEFAULT_DATA_PATH)
    else:
        return None
    dp.handle_missing_values()
    dp.remove_duplicates()
    df = dp._df
    st.session_state.df = df
    features = dp.get_feature_names()
    st.session_state.feature_names = features
    X = df[features].to_numpy().astype(float)
    y = df[dp.get_target_name()].to_numpy().astype(float)
    st.session_state.X = X
    st.session_state.y = y
    return df


def get_data_or_warn():
    if st.session_state.df is not None and st.session_state.X is not None and st.session_state.y is not None:
        return st.session_state.df, st.session_state.X, st.session_state.y, st.session_state.feature_names
    st.warning("Please load data first in the **Dataset Analysis** page.")
    return None, None, None, None


# ──────────────────────────── PAGES ────────────────────────────

def page_home():
    st.title("  Numerical Methods for House Price Prediction")
    st.markdown("---")
    st.markdown("""
This application demonstrates the application of **numerical methods** to real-world
house price prediction problems. It covers linear algebra solvers, root-finding algorithms,
interpolation, eigenvalue analysis, error analysis, numerical differentiation and
integration, ODE/BVP solvers, and regression — all implemented from scratch using
only NumPy for array operations.
""")
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Course:** Numerical Methods\n\n**Topic:** House Price Prediction using Regression & Numerical Algorithms")
    with col2:
        st.success("**Features:**\n- Multiple regression solvers (Gauss-Jordan, LU, Cholesky)\n- Root finding for price targets\n- Interpolation & sensitivity analysis\n- ODE-based price growth modelling")

    st.markdown("### Navigation")
    st.markdown("Use the **sidebar** to navigate between different sections of the application.")

    st.markdown("### Quick Start")
    st.markdown("1. Go to **Dataset Analysis** to load data\n2. Go to **Train Regression Model** to fit a model\n3. Go to **Predict House Price** to make predictions")


def page_dataset_analysis():
    st.title("  Dataset Analysis")
    st.markdown("---")

    col1, col2 = st.columns([1, 1])
    with col1:
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    with col2:
        use_default = st.checkbox("Use default house_prices.csv", value=True)

    if st.button("Load Data", type="primary"):
        df = load_data(uploaded_file=uploaded, use_default=use_default)
        if df is not None:
            st.success(f"Loaded {len(df)} records successfully.")
        else:
            st.error("No data source selected. Please upload a file or check 'Use default'.")

    df = st.session_state.df
    if df is None:
        st.info("Load data using the controls above.")
        return

    dp = st.session_state.data_processor
    summary = dp.get_summary()
    validation = dp.validate_data()

    st.subheader("Dataset Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Records", f"{summary['records']:,}")
    c2.metric("Features", summary["features"])
    c3.metric("Target", summary["target"])
    c4.metric("Missing Values", summary["missing_values"])
    c5.metric("Duplicate Rows", summary["duplicate_rows"])

    if not validation["valid"]:
        for issue in validation["issues"]:
            st.warning(issue)

    st.subheader("First 10 Rows")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Basic Statistics")
    st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Scatter Plots: Features vs Price")
    plotter = Plotter()
    features = st.session_state.feature_names
    X = st.session_state.X
    y = st.session_state.y
    if X is not None and y is not None:
        selected_feature = st.selectbox("Select feature to plot", features, index=0)
        idx = features.index(selected_feature)
        fig = plotter.dataset_scatter(X, y, features, "price", feature_index=idx)
        st.pyplot(fig)
    else:
        st.info("Data not loaded yet. Click **Load Data** above first.")


def page_train_model():
    st.title("  Train Regression Model")
    st.markdown("---")

    data_tuple = get_data_or_warn()
    if data_tuple[0] is None:
        return
    _, X, y, feature_names = data_tuple

    col1, col2 = st.columns(2)
    with col1:
        solver_choice = st.selectbox("Matrix Solver", ["gauss_jordan", "lu_decomposition", "cholesky"])
    with col2:
        regression_type = st.selectbox("Regression Type", ["Linear", "Quadratic", "Cubic"])

    if st.button("Train Model", type="primary"):
        with st.spinner("Training..."):
            engine = RegressionEngine()
            degree_map = {"Linear": 1, "Quadratic": 2, "Cubic": 3}
            degree = degree_map[regression_type]

            X_train, X_test, y_train, y_test = st.session_state.data_processor.train_test_split(X, y, test_size=0.2)

            if degree == 1:
                result = engine.fit_linear(X_train, y_train, feature_names=feature_names, solver=solver_choice)
            else:
                result = engine.fit_polynomial(X_train, y_train, degree=degree, feature_index=0, solver=solver_choice)

            if result["success"]:
                y_pred_test = engine.predict(X_test)
                y_pred_all = engine.predict(X_train)
                st.session_state.regression_engine = engine
                st.session_state.model_fitted = True
                st.session_state.coefficients = result["coefficients"]
                st.session_state.y_pred = y_pred_test
                st.session_state.y_test = y_test
                st.session_state.y_train = y_train
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_pred_train = y_pred_all
                st.session_state.solver_choice = solver_choice
                st.session_state.regression_type = regression_type
                st.success(f"Model trained using **{regression_type}** regression with **{solver_choice}** solver.")
            else:
                st.error(f"Training failed: {result.get('message', 'Unknown error')}")

    if st.session_state.model_fitted and st.session_state.coefficients is not None:
        st.subheader("Model Coefficients")
        coeffs = st.session_state.coefficients
        coeff_df = pd.DataFrame({
            "Feature": list(coeffs.keys()),
            "Coefficient": [f"{v:,.4f}" for v in coeffs.values()],
        })
        st.dataframe(coeff_df, use_container_width=True)

        st.subheader("Model Metrics")
        engine = st.session_state.regression_engine
        y_test = st.session_state.y_test
        y_pred = st.session_state.y_pred
        metrics = engine.get_model_summary(y_test, y_pred)

        c1, c2, c3 = st.columns(3)
        c1.metric("R² Score", f"{metrics['r2']:.6f}")
        c2.metric("RMSE", f"{metrics['rmse']:,.2f}")
        c3.metric("MAE", f"{metrics['mae']:,.2f}")

        st.subheader("Actual vs Predicted")
        plotter = Plotter()
        fig = plotter.actual_vs_predicted(y_test, y_pred)
        st.pyplot(fig)

        st.subheader("Residual Plot")
        fig2 = plotter.residual_plot(y_test, y_pred)
        st.pyplot(fig2)


def page_predict():
    st.title("  Predict House Price")
    st.markdown("---")

    if not st.session_state.model_fitted or st.session_state.regression_engine is None:
        st.warning("Please train a model first in **Train Regression Model**.")
        return

    st.subheader("Enter House Features")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        area = st.number_input("Area (sq ft)", min_value=500.0, max_value=10000.0, value=2500.0, step=50.0)
    with c2:
        bedrooms = st.number_input("Bedrooms", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
    with c3:
        bathrooms = st.number_input("Bathrooms", min_value=1.0, max_value=8.0, value=2.0, step=0.5)
    with c4:
        age = st.number_input("Age (years)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)

    c5, c6, c7 = st.columns(3)
    with c5:
        parking = st.number_input("Parking Spots", min_value=0.0, max_value=5.0, value=1.0, step=0.5)
    with c6:
        location_score = st.number_input("Location Score (1-10)", min_value=1.0, max_value=10.0, value=5.0, step=0.5)
    with c7:
        distance_center = st.number_input("Distance to Center (km)", min_value=0.1, max_value=50.0, value=10.0, step=0.5)

    if st.button("Predict Price", type="primary"):
        try:
            engine = st.session_state.regression_engine
            features = st.session_state.feature_names

            if st.session_state.regression_type == "Linear":
                input_arr = np.array([[area, bedrooms, bathrooms, age, parking, location_score, distance_center]])
            else:
                input_arr = np.array([[area]])

            pred = engine.predict(input_arr)
            price = float(pred[0])

            st.markdown("---")
            st.subheader("Prediction Result")
            st.metric("Predicted House Price", f"${price:,.2f}")

            st.subheader("Feature Importance")
            coeffs = st.session_state.coefficients
            coeff_vals = list(coeffs.values())[1:]
            fnames = features
            plotter = Plotter()
            fig = plotter.feature_importance(fnames, coeff_vals)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Prediction error: {e}")


def page_matrix_solver():
    st.title("  Matrix Solver")
    st.markdown("---")

    st.markdown("Solve **Ax = b** using different numerical methods. This demonstrates the solvers used internally by the regression engine.")

    source = st.radio("Data Source", ["Use regression normal equations (X^T X beta = X^T y)", "Enter manually"], horizontal=True)

    solver = MatrixSolver()

    if source == "Use regression normal equations (X^T X beta = X^T y)":
        data_tuple = get_data_or_warn()
        if data_tuple[0] is None:
            return
        _, X, y, feature_names = data_tuple

        X_design = np.hstack([np.ones((X.shape[0], 1)), X])
        A = X_design.T @ X_design
        b = X_design.T @ y
        st.info(f"System size: {A.shape[0]} x {A.shape[1]}")
    else:
        n = st.number_input("Matrix size (n)", min_value=2, max_value=10, value=3, step=1)
        st.markdown("Enter matrix A (comma-separated rows):")
        a_text = st.text_area("Matrix A", value="\n".join(["1, 2, 3"] * int(n)), height=150)
        b_text = st.text_input("Vector b (comma-separated)", value="1, 2, 3")
        try:
            rows = [list(map(float, row.split(","))) for row in a_text.strip().split("\n") if row.strip()]
            A = np.array(rows, dtype=float)
            b = np.array(list(map(float, b_text.split(","))), dtype=float)
        except Exception as e:
            st.error(f"Parse error: {e}")
            return

    method = st.selectbox("Method", ["gauss_jordan", "lu_decomposition", "cholesky", "compare_all"])

    if st.button("Solve", type="primary"):
        with st.spinner("Solving..."):
            if method == "compare_all":
                results = solver.compare_methods(A, b)
                st.subheader("Comparison Table")
                comp = results["comparison"]
                comp_df = pd.DataFrame([
                    {
                        "Method": r["method"],
                        "Success": r["success"],
                        "Residual": f"{r['residual']:.6e}" if r["residual"] is not None else "N/A",
                        "Time (s)": f"{r['execution_time']:.8f}",
                    }
                    for r in comp
                ])
                st.dataframe(comp_df, use_container_width=True)

                if results["reference_solution"] is not None:
                    st.subheader("Reference Solution")
                    sol = results["reference_solution"]
                    st.write([f"{v:.6f}" for v in sol])
            else:
                method_fn = getattr(solver, method)
                result = method_fn(A, b)

                if result["success"] and result["solution"] is not None:
                    st.success(result["message"])
                    st.subheader("Solution")
                    sol = result["solution"]
                    sol_df = pd.DataFrame({
                        "Variable": [f"x{i}" for i in range(len(sol))],
                        "Value": [f"{v:.6f}" for v in sol],
                    })
                    st.dataframe(sol_df, use_container_width=True)

                    st.subheader("Metrics")
                    c1, c2 = st.columns(2)
                    c1.metric("Residual", f"{result['residual']:.6e}")
                    c2.metric("Execution Time", f"{result['execution_time']:.8f}s")
                else:
                    st.error(result.get("message", "Solver failed"))


def page_root_finder():
    st.title("  Root Finder")
    st.markdown("---")

    st.markdown("""
Find the **area** required for a house to reach a **target price**.
Given a regression model `Price = f(Area, ...)`, we solve `f(Area, ...) - Target = 0`.
""")

    data_tuple = get_data_or_warn()
    if data_tuple[0] is None:
        return
    _, X, y, feature_names = data_tuple

    if st.session_state.model_fitted and st.session_state.coefficients is not None:
        coeffs = st.session_state.coefficients
        coeff_vals = list(coeffs.values())
    else:
        engine = RegressionEngine()
        result = engine.fit_linear(X, y, feature_names=feature_names, solver="gauss_jordan")
        coeff_vals = list(result["coefficients"].values())

    fixed_features = {}
    c1, c2, c3 = st.columns(3)
    with c1:
        fixed_features["bedrooms"] = st.number_input("Fixed Bedrooms", value=3.0, step=1.0, key="rf_bed")
    with c2:
        fixed_features["bathrooms"] = st.number_input("Fixed Bathrooms", value=2.0, step=1.0, key="rf_bath")
    with c3:
        fixed_features["age"] = st.number_input("Fixed Age", value=10.0, step=1.0, key="rf_age")
    c4, c5, c6 = st.columns(3)
    with c4:
        fixed_features["parking"] = st.number_input("Fixed Parking", value=1.0, step=1.0, key="rf_park")
    with c5:
        fixed_features["location_score"] = st.number_input("Fixed Location Score", value=5.0, step=1.0, key="rf_loc")
    with c6:
        fixed_features["distance_center"] = st.number_input("Fixed Distance", value=10.0, step=0.5, key="rf_dist")

    target_price = st.number_input("Target Price", min_value=50000, max_value=5000000, value=500000, step=50000)

    feature_map = {name: fixed_features.get(name, 0.0) for name in feature_names}
    intercept = coeff_vals[0]
    fixed_sum = sum(coeff_vals[i + 1] * feature_map[name] for i, name in enumerate(feature_names) if name != "area")
    area_idx = feature_names.index("area")
    area_coeff = coeff_vals[area_idx + 1]

    def price_minus_target(area_val):
        return intercept + area_coeff * area_val + fixed_sum - target_price

    fa_val = price_minus_target(500)
    fb_val = price_minus_target(10000)
    if fa_val * fb_val > 0:
        if area_coeff > 0:
            a_bound = 500
            while price_minus_target(a_bound) > 0 and a_bound > 1:
                a_bound = max(1, a_bound - 500)
            b_bound = 10000
            while price_minus_target(b_bound) < 0 and b_bound < 100000:
                b_bound += 5000
        else:
            a_bound = 10000
            while price_minus_target(a_bound) > 0 and a_bound < 100000:
                a_bound += 5000
            b_bound = 500
            while price_minus_target(b_bound) < 0 and b_bound > 1:
                b_bound = max(1, b_bound - 500)
        if price_minus_target(a_bound) * price_minus_target(b_bound) > 0:
            st.warning("Cannot find a root — the target price may be outside the model's range.")
            return
    else:
        a_bound = 500
        b_bound = 10000

    method = st.selectbox("Root Finding Method", ["bisection", "newton_raphson", "secant"])

    rf = RootFinder()

    if st.button("Find Required Area", type="primary"):
        with st.spinner("Finding root..."):
            if method == "bisection":
                result = rf.bisection(price_minus_target, a_bound, b_bound)
            elif method == "newton_raphson":
                df_price = lambda a: area_coeff
                result = rf.newton_raphson(price_minus_target, df_price, 2000.0)
            else:
                result = rf.secant(price_minus_target, a_bound, b_bound)

        if result["success"]:
            st.success(f"Required area: **{result['root']:,.2f} sq ft**")
            st.metric("Iterations", result["iterations"])
            st.metric("Final Error", f"{result['error']:.6e}")

            if result["convergence_history"]:
                st.subheader("Convergence Plot")
                plotter = Plotter()
                fig = plotter.root_finding_convergence(result["convergence_history"], method.replace("_", " ").title())
                st.pyplot(fig)
        else:
            st.error(result.get("message", "Root finding failed"))


def page_interpolation():
    st.title("  Interpolation")
    st.markdown("---")

    data_tuple = get_data_or_warn()
    if data_tuple[0] is None:
        return
    df, X, y, feature_names = data_tuple

    area_idx = feature_names.index("area")
    areas = X[:, area_idx]
    sort_idx = np.argsort(areas)
    areas_sorted = areas[sort_idx]
    prices_sorted = y[sort_idx]

    n_points = st.slider("Number of data points for interpolation", 5, 50, 15)
    step = max(1, len(areas_sorted) // n_points)
    idx_subset = np.arange(0, len(areas_sorted), step)[:n_points]
    x_data = areas_sorted[idx_subset]
    y_data = prices_sorted[idx_subset]

    target_area = st.number_input("Area to estimate price", min_value=float(x_data.min()), max_value=float(x_data.max()), value=float(np.median(x_data)), step=50.0)

    method = st.selectbox("Interpolation Method", ["lagrange", "newton_divided_difference", "cubic_spline", "least_squares"])

    interp = InterpolationEngine()

    if st.button("Estimate Price", type="primary"):
        results = {}
        with st.spinner("Computing..."):
            if method == "lagrange" or method == "all":
                val = interp.lagrange(x_data, y_data, target_area)
                results["Lagrange"] = val
            if method == "newton_divided_difference" or method == "all":
                r = interp.newton_divided_difference(x_data, y_data, target_area)
                results["Newton Divided Diff"] = r["value"]
            if method == "cubic_spline" or method == "all":
                r = interp.cubic_spline(x_data, y_data, target_area)
                results["Cubic Spline"] = r["value"]
            if method == "least_squares" or method == "all":
                r = interp.least_squares_fit(x_data, y_data, degree=2, x=target_area)
                results["Least Squares"] = r["value"]

        if results:
            st.subheader("Results")
            for name, val in results.items():
                st.metric(name, f"${val:,.2f}")

            st.subheader("Interpolation Curves")
            x_fine = np.linspace(x_data.min(), x_data.max(), 300)
            curves = {}
            for mname in ["Lagrange", "Newton Divided Diff", "Cubic Spline"]:
                y_fine = []
                for xf in x_fine:
                    try:
                        if mname == "Lagrange":
                            y_fine.append(interp.lagrange(x_data, y_data, xf))
                        elif mname == "Newton Divided Diff":
                            y_fine.append(interp.newton_divided_difference(x_data, y_data, xf)["value"])
                        else:
                            y_fine.append(interp.cubic_spline(x_data, y_data, xf)["value"])
                    except Exception:
                        y_fine.append(np.nan)
                curves[mname] = np.array(y_fine)

            plotter = Plotter()
            fig = plotter.interpolation_comparison(x_data, y_data, x_fine, curves)
            ax = fig.axes[0]
            ax.axvline(target_area, color="red", linestyle="--", alpha=0.7, label="Target Area")
            ax.legend()
            st.pyplot(fig)


def page_eigenvalue():
    st.title("  Eigenvalue Analysis")
    st.markdown("---")

    data_tuple = get_data_or_warn()
    if data_tuple[0] is None:
        return
    _, X, y, feature_names = data_tuple

    st.subheader("Feature Correlation Matrix")
    corr = np.corrcoef(X.T)

    corr_df = pd.DataFrame(corr, columns=feature_names, index=feature_names)
    st.dataframe(corr_df.style.format("{:.4f}"), use_container_width=True)

    eigen_solver = EigenSolver()

    if st.button("Run Eigenvalue Analysis", type="primary"):
        with st.spinner("Running Power Method..."):
            pm_result = eigen_solver.power_method(corr)

        with st.spinner("Running QR Iteration..."):
            qr_result = eigen_solver.qr_iteration(corr)

        st.subheader("Power Method (Dominant Eigenvalue)")
        if pm_result["success"]:
            st.metric("Dominant Eigenvalue", f"{pm_result['eigenvalue']:.6f}")
            st.metric("Iterations", pm_result["iterations"])
            st.metric("Time", f"{pm_result['execution_time']:.6f}s")

            ev_df = pd.DataFrame({
                "Feature": feature_names,
                "Component": [f"{v:.6f}" for v in pm_result["eigenvector"]],
            })
            st.dataframe(ev_df, use_container_width=True)

            if pm_result["convergence_history"]:
                plotter = Plotter()
                fig = plotter.eigenvalue_convergence(pm_result["convergence_history"], "Power Method")
                st.pyplot(fig)
        else:
            st.error(pm_result["message"])

        st.subheader("QR Iteration (All Eigenvalues)")
        if qr_result["success"]:
            evals = qr_result["eigenvalues"]
            evecs = qr_result["eigenvectors"]
            st.metric("Iterations", qr_result["iterations"])
            st.metric("Time", f"{qr_result['execution_time']:.6f}s")

            eval_df = pd.DataFrame({
                "Index": [f"λ{i+1}" for i in range(len(evals))],
                "Eigenvalue": [f"{v:.6f}" for v in evals],
            })
            st.dataframe(eval_df, use_container_width=True)

            evec_df = pd.DataFrame(evecs, columns=[f"λ{i+1}" for i in range(evecs.shape[1])], index=feature_names)
            st.subheader("Eigenvectors")
            st.dataframe(evec_df.style.format("{:.4f}"), use_container_width=True)

            if qr_result["convergence_history"]:
                plotter = Plotter()
                fig = plotter.eigenvalue_convergence(qr_result["convergence_history"], "QR Iteration")
                st.pyplot(fig)
        else:
            st.error(qr_result["message"])

        st.subheader("Correlation Heatmap")
        plotter = Plotter()
        fig = plotter.correlation_heatmap(corr, feature_names)
        st.pyplot(fig)


def page_error_analysis():
    st.title("  Error Analysis")
    st.markdown("---")

    if not st.session_state.model_fitted or st.session_state.regression_engine is None:
        st.warning("Please train a model first in **Train Regression Model**.")
        return

    engine = st.session_state.regression_engine
    y_test = st.session_state.y_test
    y_pred = st.session_state.y_pred
    analyzer = ErrorAnalyzer()

    st.subheader("Comprehensive Error Metrics")
    result = analyzer.prediction_error_analysis(y_test, y_pred)
    summary = result["summary"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{summary['mae']:,.2f}")
    c2.metric("RMSE", f"{summary['rmse']:,.2f}")
    c3.metric("R² Score", f"{summary['r_squared']:.6f}")
    c4.metric("Mean % Error", f"{summary['mean_percentage_error']:.4f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Mean Error", f"{summary['mean_error']:,.2f}")
    c6.metric("Std Error", f"{summary['std_error']:,.2f}")
    c7.metric("Max Abs Error", f"{summary['max_abs_error']:,.2f}")
    c8.metric("Median Abs Error", f"{summary['median_abs_error']:,.2f}")

    st.subheader("Error Distribution")
    plotter = Plotter()
    fig = plotter.error_distribution(result["individual_errors"])
    st.pyplot(fig)

    st.subheader("Residual Analysis")
    residual_result = analyzer.residual_analysis(y_test, y_pred)
    stats = residual_result["stats"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Residual Mean", f"{stats['mean']:,.2f}")
    c2.metric("Residual Std", f"{stats['std']:,.2f}")
    c3.metric("Skewness", f"{residual_result['skewness']:.4f}")
    c4.metric("Excess Kurtosis", f"{residual_result['excess_kurtosis']:.4f}")
    c5.metric("Symmetric", "Yes" if residual_result["is_symmetric"] else "No")

    st.subheader("Error Propagation Example")
    st.markdown("For **z = price_area + price_location** (addition):")
    sigma_area = st.number_input("Error in price_area", value=100000.0, step=10000.0, key="ep_a")
    sigma_loc = st.number_input("Error in price_location", value=50000.0, step=10000.0, key="ep_b")
    propagated = analyzer.error_propagation_addition([sigma_area, sigma_loc])
    st.metric("Propagated Error (addition)", f"{propagated:,.2f}")


def page_differentiation():
    st.title("  Numerical Differentiation")
    st.markdown("---")

    st.markdown("Analyze **price sensitivity** to features using numerical derivatives.")

    data_tuple = get_data_or_warn()
    if data_tuple[0] is None:
        return
    _, X, y, feature_names = data_tuple

    diff = NumericalDifferentiation()

    if st.session_state.model_fitted and st.session_state.coefficients is not None:
        coeffs = st.session_state.coefficients
        coeff_vals = list(coeffs.values())
    else:
        engine = RegressionEngine()
        result = engine.fit_linear(X, y, feature_names=feature_names, solver="gauss_jordan")
        coeff_vals = list(result["coefficients"].values())

    feature_map_vals = {}
    for i, name in enumerate(feature_names):
        feature_map_vals[name] = float(np.median(X[:, i]))

    st.subheader("Set Feature Values")
    cols = st.columns(min(4, len(feature_names)))
    for i, name in enumerate(feature_names):
        with cols[i % len(cols)]:
            feature_map_vals[name] = st.number_input(
                name, value=feature_map_vals[name], step=1.0, key=f"diff_{name}"
            )

    st.subheader("Sensitivity Analysis")
    sensitivity_data = []
    for i, fname in enumerate(feature_names):
        fixed_sum = coeff_vals[0]
        for j, fn in enumerate(feature_names):
            if fn != fname:
                fixed_sum += coeff_vals[j + 1] * feature_map_vals[fn]

        coeff_i = coeff_vals[i + 1]

        def price_fn(x, c=coeff_i, fs=fixed_sum):
            return c * x + fs

        sens = diff.price_sensitivity(price_fn, feature_map_vals[fname], fname)
        sensitivity_data.append({
            "Feature": fname,
            "Current Value": f"{feature_map_vals[fname]:,.2f}",
            "Sensitivity (dP/dF)": f"{sens['sensitivity']:,.2f}",
            "Interpretation": sens["interpretation"],
        })

    st.dataframe(pd.DataFrame(sensitivity_data), use_container_width=True)

    st.subheader("Method Comparison")
    compare_feature = st.selectbox("Select feature", feature_names, index=0)
    cidx = feature_names.index(compare_feature)
    coeff_c = coeff_vals[cidx + 1]
    fixed_sum_c = coeff_vals[0]
    for j, fn in enumerate(feature_names):
        if fn != compare_feature:
            fixed_sum_c += coeff_vals[j + 1] * feature_map_vals[fn]

    def price_fn_c(x, c=coeff_c, fs=fixed_sum_c):
        return c * x + fs

    exact_fn = lambda x: coeff_c

    comparison = diff.compare_methods(price_fn_c, exact_fn, feature_map_vals[compare_feature])
    comp_df = pd.DataFrame({
        "Method": ["Forward Difference", "Backward Difference", "Central Difference", "Exact"],
        "Result": [comparison["forward"], comparison["backward"], comparison["central"], comparison["exact"]],
        "Absolute Error": [comparison["error_forward"], comparison["error_backward"], comparison["error_central"], 0.0],
    })
    st.dataframe(comp_df, use_container_width=True)

    st.subheader("Second Derivative (Curvature)")
    second_deriv = diff.second_derivative(price_fn_c, feature_map_vals[compare_feature])
    st.metric(f"d²P/d{compare_feature}²", f"{second_deriv:,.6f}")


def page_integration():
    st.title("  Numerical Integration")
    st.markdown("---")

    st.markdown("Compute the **integral of a price function** over a range of area values.")

    func_name = st.selectbox("Price Function", ["Linear: P(a) = 5000*a + 1000000", "Quadratic: P(a) = 0.5*a² + 3000*a + 500000", "Cubic: P(a) = 0.001*a³ + 2*a² + 1000*a"])

    if "Linear" in func_name:
        f = lambda a: 5000 * a + 1000000
        exact = lambda a, b: 2500 * (b**2 - a**2) + 1000000 * (b - a)
    elif "Quadratic" in func_name:
        f = lambda a: 0.5 * a**2 + 3000 * a + 500000
        exact = lambda a, b: (0.5 / 3) * (b**3 - a**3) + 1500 * (b**2 - a**2) + 500000 * (b - a)
    else:
        f = lambda a: 0.001 * a**3 + 2 * a**2 + 1000 * a
        exact = lambda a, b: (0.001 / 4) * (b**4 - a**4) + (2.0 / 3) * (b**3 - a**3) + 500 * (b**2 - a**2)

    c1, c2 = st.columns(2)
    with c1:
        a_val = st.number_input("Lower bound (area)", value=500.0, step=50.0)
    with c2:
        b_val = st.number_input("Upper bound (area)", value=3000.0, step=50.0)

    if st.button("Compute Integral", type="primary"):
        integration = NumericalIntegration()
        exact_val = exact(a_val, b_val)

        with st.spinner("Computing..."):
            trap = integration.trapezoidal(f, a_val, b_val)
            s13 = integration.simpson_one_third(f, a_val, b_val)
            s38 = integration.simpson_three_eighth(f, a_val, b_val)
            g2 = integration.gaussian_quadrature_2point(f, a_val, b_val)
            g3 = integration.gaussian_quadrature_3point(f, a_val, b_val)

        st.subheader("Results")
        results_df = pd.DataFrame({
            "Method": ["Trapezoidal", "Simpson's 1/3", "Simpson's 3/8", "Gaussian 2-point", "Gaussian 3-point", "Exact"],
            "Result": [trap["result"], s13["result"], s38["result"], g2["result"], g3["result"], exact_val],
            "Absolute Error": [
                abs(exact_val - trap["result"]),
                abs(exact_val - s13["result"]),
                abs(exact_val - s38["result"]),
                abs(exact_val - g2["result"]),
                abs(exact_val - g3["result"]),
                0.0,
            ],
        })
        st.dataframe(results_df, use_container_width=True)

        st.subheader("Bar Chart Comparison")
        plotter = Plotter()
        chart_data = {
            "Trapezoidal": trap["result"],
            "Simpson's 1/3": s13["result"],
            "Simpson's 3/8": s38["result"],
            "Gaussian 2-pt": g2["result"],
            "Gaussian 3-pt": g3["result"],
            "Exact": exact_val,
        }
        fig = plotter.integration_comparison(chart_data)
        st.pyplot(fig)


def page_ode_solver():
    st.title("  ODE Solver")
    st.markdown("---")

    st.markdown("""
**House Price Growth Model:** dP/dt = r * P

Where P is price, t is time (years), and r is the growth rate.
Exact solution: P(t) = P0 * exp(r * t)
""")

    c1, c2, c3 = st.columns(3)
    with c1:
        P0 = st.number_input("Initial Price P₀", min_value=100000, max_value=100000000, value=10000000, step=500000)
    with c2:
        r = st.number_input("Growth Rate r", min_value=-0.5, max_value=1.0, value=0.05, step=0.01, format="%.3f")
    with c3:
        t_end = st.number_input("Time Span (years)", min_value=1, max_value=50, value=10, step=1)

    n_steps = st.slider("Number of steps", 10, 500, 100)

    if st.button("Solve ODE", type="primary"):
        ode_solver = ODESolver()
        f = lambda t, y: r * y
        t_span = (0.0, float(t_end))
        exact_solution = lambda t: P0 * np.exp(r * t)

        with st.spinner("Solving..."):
            comparison = ode_solver.compare_methods(f, t_span, float(P0), n_steps, exact_solution)

        rk4 = comparison["rk4"]
        ab4 = comparison["adams_bashforth_4"]

        st.subheader("Results Comparison")
        step = max(1, len(rk4["t_values"]) // 20)
        display_idx = np.arange(0, len(rk4["t_values"]), step)

        results_df = pd.DataFrame({
            "Time (years)": [f"{rk4['t_values'][i]:.2f}" for i in display_idx],
            "RK4": [f"{rk4['y_values'][i]:,.2f}" for i in display_idx],
            "Adams-Bashforth": [f"{ab4['y_values'][i]:,.2f}" for i in display_idx],
            "Exact": [f"{comparison['exact_values'][i]:,.2f}" for i in display_idx],
        })
        st.dataframe(results_df, use_container_width=True)

        c1, c2 = st.columns(2)
        c1.metric("RK4 Max Error", f"{comparison['rk4_max_error']:,.2f}")
        c2.metric("Adams-Bashforth Max Error", f"{comparison['ab4_max_error']:,.2f}")

        st.subheader("Solution Plot")
        plotter = Plotter()
        fig = plotter.ode_solution(
            rk4["t_values"],
            {"RK4": rk4["y_values"], "Adams-Bashforth 4": ab4["y_values"], "Exact": comparison["exact_values"]},
        )
        st.pyplot(fig)


def page_bvp_solver():
    st.title("  BVP Solver")
    st.markdown("---")

    st.markdown("""
**Boundary Value Problem:** Solve y'' + p(x)*y' + q(x)*y = r(x)

Boundary conditions: y(a) = alpha, y(b) = beta

Demonstrates the finite difference and shooting methods.
""")

    preset = st.selectbox("Problem Preset", ["Heat distribution: y'' = -π²sin(πx)", "Custom"])

    if preset.startswith("Heat"):
        p_func = lambda x: 0.0
        q_func = lambda x: 0.0
        r_func = lambda x: -np.pi**2 * np.sin(np.pi * x)
        a, b = 0.0, 1.0
        alpha, beta = 0.0, 0.0
        f_shooting = lambda x, y, dy: -np.pi**2 * np.sin(np.pi * x)
    else:
        st.markdown("Enter coefficients:")
        p_val = st.number_input("p(x) constant value", value=0.0, step=0.1)
        q_val = st.number_input("q(x) constant value", value=0.0, step=0.1)
        r_val = st.number_input("r(x) constant value", value=-5.0, step=0.5)
        a = st.number_input("a (left boundary)", value=0.0, step=0.1)
        b = st.number_input("b (right boundary)", value=1.0, step=0.1)
        alpha = st.number_input("y(a)", value=0.0, step=0.1)
        beta = st.number_input("y(b)", value=0.0, step=0.1)

        p_func = lambda x: p_val
        q_func = lambda x: q_val
        r_func = lambda x: r_val
        f_shooting = lambda x, y, dy: -p_val * dy - q_val * y + r_val

    n_fd = st.slider("Grid points (finite difference)", 5, 100, 20)
    n_shooting = st.slider("Steps (shooting method)", 10, 500, 100)

    if st.button("Solve BVP", type="primary"):
        bvp = BVPSolver()

        with st.spinner("Solving with Finite Difference..."):
            fd_result = bvp.finite_difference(p_func, q_func, r_func, a, b, alpha, beta, n=n_fd)

        with st.spinner("Solving with Shooting Method..."):
            shoot_result = bvp.shooting_method(f_shooting, a, b, alpha, beta, n_steps=n_shooting)

        st.subheader("Finite Difference Solution")
        st.write(f"Points: {fd_result['n_points']}")
        st.dataframe(pd.DataFrame({
            "x": fd_result["x_values"],
            "y(x)": fd_result["y_values"],
        }), use_container_width=True)

        st.subheader("Shooting Method Solution")
        st.write(f"Initial slope found: {shoot_result['initial_slope']:.6f}")

        st.subheader("Comparison Plot")
        fig, ax = __import__("matplotlib").pyplot.subplots(figsize=(10, 6))
        ax.plot(fd_result["x_values"], fd_result["y_values"], "b-o", markersize=3, label="Finite Difference", linewidth=2)
        ax.plot(shoot_result["x_values"], shoot_result["y_values"], "r--", markersize=3, label="Shooting Method", linewidth=2)
        ax.set_xlabel("x")
        ax.set_ylabel("y(x)")
        ax.set_title("BVP Solution Comparison")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)


def page_method_comparison():
    st.title("  Method Comparison")
    st.markdown("---")

    data_tuple = get_data_or_warn()
    if data_tuple[0] is None:
        return
    _, X, y, feature_names = data_tuple

    tab1, tab2, tab3, tab4 = st.tabs(["Matrix Solvers", "Root Finding", "Interpolation", "Integration"])

    with tab1:
        st.subheader("Matrix Solver Comparison for Regression")
        if st.button("Compare Matrix Solvers", type="primary", key="cmp_matrix"):
            engine = RegressionEngine()
            X_train, X_test, y_train, y_test = st.session_state.data_processor.train_test_split(X, y, test_size=0.2)

            comparison_rows = []
            for solver_name in ["gauss_jordan", "lu_decomposition", "cholesky"]:
                try:
                    eng = RegressionEngine()
                    result = eng.fit_linear(X_train, y_train, feature_names=feature_names, solver=solver_name)
                    if result["success"]:
                        y_pred = eng.predict(X_test)
                        r2 = eng.calculate_r2(y_test, y_pred)
                        rmse = eng.calculate_rmse(y_test, y_pred)
                        mae = eng.calculate_mae(y_test, y_pred)
                        comparison_rows.append({
                            "Solver": solver_name,
                            "R²": f"{r2:.6f}",
                            "RMSE": f"{rmse:,.2f}",
                            "MAE": f"{mae:,.2f}",
                            "Residual": f"{result['residual']:.6e}",
                            "Success": True,
                        })
                    else:
                        comparison_rows.append({"Solver": solver_name, "Success": False, "R²": "N/A", "RMSE": "N/A", "MAE": "N/A", "Residual": "N/A"})
                except Exception as e:
                    comparison_rows.append({"Solver": solver_name, "Success": False, "R²": str(e), "RMSE": "N/A", "MAE": "N/A", "Residual": "N/A"})

            st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True)

    with tab2:
        st.subheader("Root Finding Method Comparison")
        target = st.number_input("Target Price for comparison", value=20000000, step=1000000, key="cmp_rf")

        if st.session_state.model_fitted and st.session_state.coefficients is not None:
            coeffs = st.session_state.coefficients
            coeff_vals = list(coeffs.values())
        else:
            engine = RegressionEngine()
            result = engine.fit_linear(X, y, feature_names=feature_names, solver="gauss_jordan")
            coeff_vals = list(result["coefficients"].values())

        area_idx = feature_names.index("area")
        area_coeff = coeff_vals[area_idx + 1]
        fixed_vals = {}
        for i, fn in enumerate(feature_names):
            fixed_vals[fn] = float(np.median(X[:, feature_names.index(fn)]))
        intercept = coeff_vals[0]
        fixed_sum = sum(coeff_vals[i + 1] * fixed_vals[fn] for i, fn in enumerate(feature_names) if fn != "area")

        def price_fn(area_val):
            return intercept + area_coeff * area_val + fixed_sum - target

        if st.button("Compare Root Finders", type="primary", key="cmp_rf_btn"):
            rf = RootFinder()
            rows = []
            for method in ["bisection", "newton_raphson", "secant"]:
                try:
                    if method == "bisection":
                        res = rf.bisection(price_fn, 500, 10000)
                    elif method == "newton_raphson":
                        res = rf.newton_raphson(price_fn, lambda a: area_coeff, 2000.0)
                    else:
                        res = rf.secant(price_fn, 1000.0, 5000.0)
                    rows.append({
                        "Method": method,
                        "Root (Area)": f"{res['root']:,.2f}" if res["root"] else "N/A",
                        "Iterations": res["iterations"],
                        "Error": f"{res['error']:.6e}" if res["error"] else "N/A",
                        "Success": res["success"],
                    })
                except Exception as e:
                    rows.append({"Method": method, "Root (Area)": str(e), "Iterations": 0, "Error": "N/A", "Success": False})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with tab3:
        st.subheader("Interpolation Method Comparison")
        area_idx = feature_names.index("area")
        areas = X[:, area_idx]
        sort_idx = np.argsort(areas)
        areas_s = areas[sort_idx]
        prices_s = y[sort_idx]
        n_pts = 10
        step = max(1, len(areas_s) // n_pts)
        idx_sub = np.arange(0, len(areas_s), step)[:n_pts]
        xd = areas_s[idx_sub]
        yd = prices_s[idx_sub]

        test_area = st.number_input("Test area", value=float(np.median(xd)), key="cmp_int")

        if st.button("Compare Interpolation", type="primary", key="cmp_int_btn"):
            interp = InterpolationEngine()
            rows = []
            for mname in ["Lagrange", "Newton", "Cubic Spline", "Least Squares"]:
                try:
                    if mname == "Lagrange":
                        val = interp.lagrange(xd, yd, test_area)
                    elif mname == "Newton":
                        val = interp.newton_divided_difference(xd, yd, test_area)["value"]
                    elif mname == "Cubic Spline":
                        val = interp.cubic_spline(xd, yd, test_area)["value"]
                    else:
                        val = interp.least_squares_fit(xd, yd, 2, test_area)["value"]
                    rows.append({"Method": mname, "Estimated Price": f"${val:,.2f}"})
                except Exception as e:
                    rows.append({"Method": mname, "Estimated Price": str(e)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    with tab4:
        st.subheader("Numerical Integration Comparison")
        func_choice = st.selectbox("Function", ["f(x) = x²", "f(x) = sin(x)", "f(x) = e^x"], key="cmp_int_func")
        a_int = st.number_input("a", value=0.0, key="cmp_int_a")
        b_int = st.number_input("b", value=1.0, key="cmp_int_b")

        if func_choice == "f(x) = x²":
            f_int = lambda x: x**2
            exact_int = (b_int**3 - a_int**3) / 3.0
        elif func_choice == "f(x) = sin(x)":
            f_int = lambda x: np.sin(x)
            exact_int = -np.cos(b_int) + np.cos(a_int)
        else:
            f_int = lambda x: np.exp(x)
            exact_int = np.exp(b_int) - np.exp(a_int)

        if st.button("Compare Integration", type="primary", key="cmp_int_btn2"):
            integration = NumericalIntegration()
            results = integration.compare_methods(f_int, a_int, b_int, exact=exact_int)

            rows = []
            for method_name, val in results.items():
                if method_name == "errors":
                    continue
                if method_name == "exact":
                    continue
                err = results.get("errors", {}).get(method_name, "N/A")
                rows.append({
                    "Method": method_name,
                    "Result": f"{val:.8f}",
                    "Absolute Error": f"{err:.8e}" if isinstance(err, float) else "N/A",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            st.metric("Exact Value", f"{exact_int:.8f}")


# ──────────────────────────── MAIN ────────────────────────────

def main():
    st.sidebar.title("  Navigation")
    st.sidebar.markdown("---")

    pages = [
        "Home",
        "Dataset Analysis",
        "Train Regression Model",
        "Predict House Price",
        "Matrix Solver",
        "Root Finder",
        "Interpolation",
        "Eigenvalue Analysis",
        "Error Analysis",
        "Numerical Differentiation",
        "Numerical Integration",
        "ODE Solver",
        "BVP Solver",
        "Method Comparison",
    ]

    selected = st.sidebar.radio("Go to", pages, index=pages.index(st.session_state.page))
    st.session_state.page = selected

    st.sidebar.markdown("---")
    if st.session_state.df is not None:
        st.sidebar.success(f"Data loaded: {len(st.session_state.df)} records")
    else:
        st.sidebar.info("No data loaded")
    if st.session_state.model_fitted:
        st.sidebar.success("Model trained")
    else:
        st.sidebar.info("No model trained")

    page_map = {
        "Home": page_home,
        "Dataset Analysis": page_dataset_analysis,
        "Train Regression Model": page_train_model,
        "Predict House Price": page_predict,
        "Matrix Solver": page_matrix_solver,
        "Root Finder": page_root_finder,
        "Interpolation": page_interpolation,
        "Eigenvalue Analysis": page_eigenvalue,
        "Error Analysis": page_error_analysis,
        "Numerical Differentiation": page_differentiation,
        "Numerical Integration": page_integration,
        "ODE Solver": page_ode_solver,
        "BVP Solver": page_bvp_solver,
        "Method Comparison": page_method_comparison,
    }

    page_map[selected]()


if __name__ == "__main__":
    main()
