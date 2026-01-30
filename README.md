# MILLOPT Pro 🏭

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Advanced Rolling Simulation & Optimization Suite with Temperature-Dependent Material Modeling**

A comprehensive web-based rolling mill simulator that provides real-time analysis of hot rolling processes. Built with Streamlit and powered by advanced metallurgical models including Von Kármán pressure distribution, Zener-Hollomon flow stress equations, and temperature-dependent material properties.

---

## 🌟 Key Features

### 🔬 Advanced Physics Models
- **Von Kármán Roll Pressure Distribution** - Accurate pressure profile calculation considering friction hill
- **Zener-Hollomon Flow Stress Model** - Temperature and strain rate dependent material behavior
- **Hitchcock Roll Flattening** - Elastic deformation of rolls under load
- **Temperature-Dependent Friction** - Dynamic friction modeling based on temperature, speed, and roll material
- **Multi-Material Support** - SG Iron and DPIC roll materials with distinct thermal properties
- **Steel Grade Database** - Low-C, IF, and Medium-C steels with calibrated constitutive parameters

### 📊 Comprehensive Analysis Tools
- **Real-time Calculations** - Rolling force, torque, power consumption, and efficiency metrics
- **Sensitivity Analysis** - Temperature, speed, and friction effect visualization
- **Material Comparison** - Side-by-side performance analysis of different roll materials
- **Process Optimization** - Multi-objective parameter optimization
- **Advanced Diagnostics** - Thermal analysis, strain rate distribution, and energy efficiency

### 🎨 Modern UI/UX
- **Dark Industrial Theme** - Professional, easy-on-the-eyes interface
- **Interactive Dashboards** - Plotly-powered dynamic visualizations
- **Modular Tab System** - Organized workflow for different analysis modes
- **KPI Cards** - Key metrics displayed prominently
- **PDF Report Generation** - Export results for documentation and review

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/millopt-pro.git
cd millopt-pro
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Run the application**
```bash
streamlit run rolling_simulator_improved.py
```

4. **Access the app**
Open your browser and navigate to `http://localhost:8501`

---

## 📋 Requirements

```
streamlit
plotly
numpy
pandas
fpdf
```

Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## 🎯 Usage Guide

### 1️⃣ **Configure Process Parameters**

Use the sidebar to set up your rolling scenario:

- **Roll Material**: Choose between SG Iron or DPIC
- **Steel Grade**: Select Low-C Steel, IF Steel, or Medium-C Steel
- **Geometric Parameters**: 
  - Initial thickness (h₁)
  - Final thickness (h₂)
  - Roll diameter (D)
  - Strip width (b)
- **Process Conditions**:
  - Roll speed (rpm)
  - Rolling temperature (°C)
  - Initial roll temperature (°C)
- **Friction Model**: Temperature-dependent, constant, or speed-dependent

### 2️⃣ **Analyze Results**

#### Dashboard Tab 📊
- View rolling load, torque, and power consumption
- Analyze temperature effects on process parameters
- Study friction coefficient impact
- Examine speed sensitivity

#### Material Compare Tab 🆚
- Side-by-side comparison of SG Iron vs DPIC
- Performance metrics across temperature ranges
- Material selection recommendations
- Cost-benefit analysis

#### Optimize Tab 🧠
- Multi-objective optimization
- Minimize power consumption
- Maximize throughput
- Balance load and efficiency

#### Advanced Analysis Tab 📈
- Detailed thermal analysis
- Strain rate distributions
- Energy efficiency metrics
- Roll temperature profiles

### 3️⃣ **Export Results**

Generate comprehensive PDF reports with:
- Process parameters summary
- Calculated metrics (force, torque, power)
- Timestamp and configuration details
- Material and friction model used

---

## 🧮 Technical Details

### Core Calculation Models

#### Flow Stress (Zener-Hollomon Model)
```
σ̄ = (1/α) × asinh[(Z/A)^(1/n)]
Z = ε̇ × exp(Q/RT)
```

Where:
- `σ̄` = Mean flow stress (MPa)
- `Z` = Zener-Hollomon parameter
- `ε̇` = Strain rate (s⁻¹)
- `Q` = Activation energy (J/mol)
- `R` = Gas constant (8.314 J/mol·K)
- `T` = Temperature (K)

#### Von Kármán Roll Pressure
```
p_mean = σ̄ × (1 + m × L / h_avg)
```

Where:
- `p_mean` = Mean roll pressure
- `m` = Friction factor
- `L` = Contact length
- `h_avg` = Average thickness

#### Roll Flattening (Hitchcock Formula)
```
R_eff = R₀ + F / (π × b × E')
E' = E / (1 - ν²)
```

Where:
- `R_eff` = Effective roll radius
- `R₀` = Nominal roll radius
- `F` = Rolling force
- `E` = Young's modulus
- `ν` = Poisson's ratio

#### Temperature-Dependent Friction
```
μ = μ₀ + a × (T - 900) - b × ln(v)
μ = μ × (k₀ / k_roll)
```

Accounts for:
- Temperature effects on lubrication
- Speed-dependent hydrodynamic effects
- Roll material thermal conductivity

---

## 📐 Material Properties Database

### Roll Materials

| Property | SG Iron | DPIC |
|----------|---------|------|
| Young's Modulus (GPa) | 170 | 190 |
| Poisson's Ratio | 0.28 | 0.28 |
| Thermal Conductivity (W/m·K) | 40.0 | 32.0 |
| Brinell Hardness | 200 | 450 |
| Density (kg/m³) | 7200 | 7500 |
| Wear Resistance | 1.0 | 1.8 |

### Steel Grades

| Grade | A (s⁻¹) | n | α (MPa⁻¹) | Q (kJ/mol) | T_recryst (°C) |
|-------|---------|---|-----------|------------|----------------|
| Low-C Steel | 1.0×10¹³ | 5.0 | 0.012 | 320 | 850 |
| IF Steel | 5.0×10¹² | 4.8 | 0.011 | 300 | 820 |
| Medium-C Steel | 2.0×10¹³ | 5.5 | 0.013 | 340 | 880 |

---

## 📊 Output Metrics

The simulator calculates and displays:

### Primary Outputs
- **Rolling Force (kN)** - Total separating force between rolls
- **Torque (kN·m)** - Required torque per roll
- **Power (kW)** - Total power consumption
- **Mean Roll Pressure (MPa)** - Average contact pressure

### Secondary Metrics
- **Contact Length (mm)** - Arc of contact
- **Roll Flattening (mm)** - Elastic deformation magnitude
- **Friction Coefficient (μ)** - Dynamic friction value
- **Flow Stress (MPa)** - Material resistance at temperature
- **Strain Rate (s⁻¹)** - Deformation rate
- **Exit Temperature (°C)** - Strip temperature after rolling
- **Roll Surface Temperature (°C)** - Operating roll temperature
- **Specific Energy (kJ/m²)** - Energy efficiency metric

---

## 🎨 Screenshots

### Main Dashboard
![Dashboard](screenshots/dashboard.png)
*Real-time rolling parameter analysis with KPI cards and sensitivity plots*

### Material Comparison
![Comparison](screenshots/comparison.png)
*Side-by-side performance analysis of different roll materials*

### Optimization Module
![Optimization](screenshots/optimization.png)
*Multi-objective parameter optimization interface*

---

## 🔬 Validation

This simulator has been validated against:
- ✅ Published research in rolling mill theory
- ✅ Industrial hot rolling process data
- ✅ Standard metallurgical reference equations
- ✅ Temperature-dependent material behavior models

See `MILLOPT_Pro_Report_*.pdf` for detailed validation reports.

---

## 🛠️ Project Structure

```
Rolling-Ecell/
├── rolling_simulator_improved.py    # Main application (v2.0)
├── rolling_simulator.py             # Previous version
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── MILLOPT_Pro_Report_*.pdf         # Validation reports
└── roll_log.csv                     # Process data logs
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 To-Do / Roadmap

- [ ] Add multi-pass rolling simulation
- [ ] Implement work roll thermal crown calculation
- [ ] Add strip profile and flatness prediction
- [ ] Include roll wear prediction models
- [ ] Support for cold rolling processes
- [ ] Database integration for historical data tracking
- [ ] REST API for external integrations
- [ ] Machine learning-based parameter recommendations

---

## 📚 References

### Rolling Theory
1. **Von Kármán, T.** (1925) - "On the theory of rolling"
2. **Hitchcock, J.H.** (1935) - "Roll Neck Bearings" (Roll flattening)
3. **Ekelund, S.** (1933) - "The analysis of factors influencing rolling pressure and power consumption in the hot rolling of steel"

### Material Modeling
4. **Sellars, C.M. & Tegart, W.J.McG.** (1972) - "Hot Workability" (Zener-Hollomon)
5. **Roberts, W.L.** (1983) - "Hot Rolling of Steel" (Flow stress models)

### Industrial Applications
6. **Ginzburg, V.B.** (1989) - "Steel-Rolling Technology: Theory and Practice"
7. **Lenard, J.G.** (2007) - "Primer on Flat Rolling"

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Authors

**Clairvyn AI/ML Team**
- Rolling Mill Simulation & Optimization
- Advanced Material Modeling
- Industrial Process Analytics

---

## 🙏 Acknowledgments

- **Streamlit** - For the excellent web framework
- **Plotly** - For interactive visualization capabilities
- **NumPy** - For numerical computations
- **The metallurgical community** - For published research and validation data

---

## 📧 Contact

For questions, suggestions, or collaboration opportunities:
- 📧 Email: your.email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/millopt-pro/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/millopt-pro/discussions)

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/millopt-pro&type=Date)](https://star-history.com/#yourusername/millopt-pro&Date)

---

<div align="center">

**Made with ❤️ for the rolling mill industry**

[Report Bug](https://github.com/yourusername/millopt-pro/issues) · [Request Feature](https://github.com/yourusername/millopt-pro/issues) · [Documentation](https://github.com/yourusername/millopt-pro/wiki)

</div>
