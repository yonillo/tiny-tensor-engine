use crate::quant::QuantizedTensorQ4Golden;
use crate::tensor::Tensor2D;
use std::fs::File;
use std::io::Write;
use std::path::Path;

/// Transformación de Yeo-Johnson (AEDV - utilidades.R / ModeloMemoriaProyectoPersonalAEDV.Rmd)
/// Normaliza distribuciones asimétricas y de colas pesadas sobre todo R (-inf, +inf).
#[inline]
pub fn yeo_johnson(y: f32, lambda: f32) -> f32 {
    if y >= 0.0 {
        if lambda.abs() < 1e-5 {
            (y + 1.0).ln()
        } else {
            ((y + 1.0).powf(lambda) - 1.0) / lambda
        }
    } else {
        let two_minus_l = 2.0 - lambda;
        if two_minus_l.abs() < 1e-5 {
            -((-y + 1.0).ln())
        } else {
            -(((-y + 1.0).powf(two_minus_l) - 1.0) / two_minus_l)
        }
    }
}

/// Transformación inversa analítica de Yeo-Johnson: recupera el peso original `y` desde el espacio normalizado `z`.
#[inline]
pub fn yeo_johnson_inverse(z: f32, lambda: f32) -> f32 {
    if z >= 0.0 {
        if lambda.abs() < 1e-5 {
            z.exp() - 1.0
        } else {
            (lambda * z + 1.0).max(1e-7).powf(1.0 / lambda) - 1.0
        }
    } else {
        let two_minus_l = 2.0 - lambda;
        if two_minus_l.abs() < 1e-5 {
            1.0 - (-z).exp()
        } else {
            1.0 - (-(two_minus_l * z) + 1.0).max(1e-7).powf(1.0 / two_minus_l)
        }
    }
}

/// Estima el parámetro `lambda*` óptimo de Yeo-Johnson minimizando la asimetría al cuadrado
/// (`skewness^2`) mediante Búsqueda de la Sección Dorada (uniendo AEDV + OH Algoritmo 4).
pub fn optimize_yeo_johnson_lambda(samples: &[f32]) -> f32 {
    const GOLDEN_R: f32 = 0.618_034;
    let step = (samples.len() / 4096).max(1);

    let eval_skew_loss = |lam: f32| -> f64 {
        let mut sum = 0.0f64;
        let mut count = 0usize;
        for i in (0..samples.len()).step_by(step) {
            sum += yeo_johnson(samples[i] * 20.0, lam) as f64;
            count += 1;
        }
        let mean = sum / count as f64;
        let mut m2 = 0.0f64;
        let mut m3 = 0.0f64;
        for i in (0..samples.len()).step_by(step) {
            let d = (yeo_johnson(samples[i] * 20.0, lam) as f64) - mean;
            let d2 = d * d;
            m2 += d2;
            m3 += d2 * d;
        }
        let var = (m2 / count as f64).max(1e-12);
        let skew = (m3 / count as f64) / var.powf(1.5);
        skew * skew
    };

    let mut a = 0.20f32;
    let mut b = 1.80f32;
    let mut x1 = b - GOLDEN_R * (b - a);
    let mut x2 = a + GOLDEN_R * (b - a);
    let mut f1 = eval_skew_loss(x1);
    let mut f2 = eval_skew_loss(x2);

    for _ in 0..12 {
        if f1 < f2 {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = b - GOLDEN_R * (b - a);
            f1 = eval_skew_loss(x1);
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + GOLDEN_R * (b - a);
            f2 = eval_skew_loss(x2);
        }
    }
    0.5 * (a + b)
}

/// Perfil Estadístico Exploratorio Multidimensional (AEDV: Asimetría, Curtosis, IQR, Yeo-Johnson).
#[derive(Clone, Debug)]
pub struct AedvTensorProfile {
    pub name: String,
    pub rows: usize,
    pub cols: usize,
    pub mean: f64,
    pub std_dev: f64,
    pub median: f32,
    pub iqr: f32,
    pub min: f32,
    pub max: f32,
    pub skewness: f64,
    pub excess_kurtosis: f64,
    pub outlier_3sigma_pct: f64,
    pub yeo_johnson_lambda: f32,
}

impl AedvTensorProfile {
    pub fn analyze(name: &str, tensor: &Tensor2D) -> Self {
        let n = tensor.data.len() as f64;
        let mean = tensor.data.iter().map(|&x| x as f64).sum::<f64>() / n;

        let mut m2 = 0.0f64;
        let mut m3 = 0.0f64;
        let mut m4 = 0.0f64;
        let mut min_v = f32::INFINITY;
        let mut max_v = f32::NEG_INFINITY;

        for &x in &tensor.data {
            if x < min_v {
                min_v = x;
            }
            if x > max_v {
                max_v = x;
            }
            let d = (x as f64) - mean;
            let d2 = d * d;
            m2 += d2;
            m3 += d2 * d;
            m4 += d2 * d2;
        }

        let var = (m2 / n).max(1e-15);
        let std_dev = var.sqrt();
        let skewness = (m3 / n) / var.powf(1.5);
        let excess_kurtosis = (m4 / n) / (var * var) - 3.0;

        let threshold_3s = 3.0 * std_dev;
        let outliers = tensor
            .data
            .iter()
            .filter(|&&x| ((x as f64) - mean).abs() > threshold_3s)
            .count();
        let outlier_3sigma_pct = (outliers as f64 / n) * 100.0;

        // Quantiles (Q1, Median, Q3) for Tukey IQR
        let mut sorted = tensor.data.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let len = sorted.len();
        let q1 = sorted[len / 4];
        let median = sorted[len / 2];
        let q3 = sorted[(3 * len) / 4];
        let iqr = q3 - q1;

        let yeo_johnson_lambda = optimize_yeo_johnson_lambda(&tensor.data);

        Self {
            name: name.to_string(),
            rows: tensor.rows,
            cols: tensor.cols,
            mean,
            std_dev,
            median,
            iqr,
            min: min_v,
            max: max_v,
            skewness,
            excess_kurtosis,
            outlier_3sigma_pct,
            yeo_johnson_lambda,
        }
    }
}

/// Counts how many blocks selected each of the 4 hardware modes (`00, 01, 10, 11`) in `Q4_GoldenOpt`.
pub fn analyze_golden_modes(q: &QuantizedTensorQ4Golden) -> [usize; 4] {
    let mut counts = [0usize; 4];
    for blk in &q.blocks {
        let m0 = (blk.scale_bf16 >> 15) as usize;
        let m1 = (blk.alpha_bf16 >> 15) as usize;
        let mode = (m1 << 1) | m0;
        counts[mode & 3] += 1;
    }
    counts
}

/// Result row for a single real LLM tensor benchmark across all quantization methods.
pub struct RealLayerBenchmarkRow {
    pub profile: AedvTensorProfile,
    pub sqnr_q8: f32,
    pub sqnr_q4_0: f32,
    pub sqnr_q4_had: f32,
    pub sqnr_q4_stats: f32,
    pub sqnr_q4_gold: f32,
    pub mode_counts: [usize; 4],
}

/// Generates an interactive Plotly HTML Dashboard (Tema 7 de AEDV: Cuadro de Mandos Interactivo)
/// visualizing the exploratory data analysis and quantization benchmarks on real `SmolLM2-135M` weights.
pub fn generate_aedv_dashboard<P: AsRef<Path>>(
    output_path: P,
    rows: &[RealLayerBenchmarkRow],
    _cascade_sqnr: [f64; 4],
    _cascade_cos: [f64; 4],
    cascade_top1: [f64; 4],
) -> Result<(), String> {
    if let Some(parent) = output_path.as_ref().parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }

    let labels: Vec<String> = rows.iter().map(|r| format!("'{}'", r.profile.name)).collect();
    let labels_js = labels.join(", ");

    let q4_0_js = rows
        .iter()
        .map(|r| format!("{:.2}", r.sqnr_q4_0))
        .collect::<Vec<_>>()
        .join(", ");
    let q4_had_js = rows
        .iter()
        .map(|r| format!("{:.2}", r.sqnr_q4_had))
        .collect::<Vec<_>>()
        .join(", ");
    let q4_stats_js = rows
        .iter()
        .map(|r| format!("{:.2}", r.sqnr_q4_stats))
        .collect::<Vec<_>>()
        .join(", ");
    let q4_gold_js = rows
        .iter()
        .map(|r| format!("{:.2}", r.sqnr_q4_gold))
        .collect::<Vec<_>>()
        .join(", ");

    let kurt_js = rows
        .iter()
        .map(|r| format!("{:.2}", r.profile.excess_kurtosis))
        .collect::<Vec<_>>()
        .join(", ");

    let mut total_modes = [0usize; 4];
    for r in rows {
        for m in 0..4 {
            total_modes[m] += r.mode_counts[m];
        }
    }

    let mut table_rows_html = String::new();
    for r in rows {
        let p = &r.profile;
        table_rows_html.push_str(&format!(
            "<tr>
                <td><code>{}</code></td>
                <td>{} × {}</td>
                <td>{:.4}</td>
                <td>{:.4}</td>
                <td><b>{:.2}</b></td>
                <td>{:.3}%</td>
                <td>{:.3}</td>
                <td>{:.2} dB</td>
                <td>{:.2} dB</td>
                <td>{:.2} dB</td>
                <td style='color:#38bdf8;font-weight:bold'>{:.2} dB (+{:.2} dB)</td>
            </tr>",
            p.name,
            p.rows,
            p.cols,
            p.std_dev,
            p.iqr,
            p.excess_kurtosis,
            p.outlier_3sigma_pct,
            p.yeo_johnson_lambda,
            r.sqnr_q4_0,
            r.sqnr_q4_had,
            r.sqnr_q4_stats,
            r.sqnr_q4_gold,
            r.sqnr_q4_gold - r.sqnr_q4_0
        ));
    }

    let html = format!(
        r#"<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>AEDV Interactive Telemetry Dashboard — Real SmolLM2-135M Weights</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  body {{ background: #0f172a; color: #e2e8f0; font-family: 'Segoe UI', system-ui, sans-serif; margin: 0; padding: 24px; }}
  h1, h2, h3 {{ color: #f8fafc; margin-top: 0; }}
  .header {{ background: linear-gradient(135deg, #1e293b, #0f172a); border: 1px solid #334155; border-radius: 12px; padding: 20px 28px; margin-bottom: 24px; }}
  .badge {{ display: inline-block; background: #0284c7; color: white; padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; margin-right: 8px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }}
  .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 18px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13.5px; }}
  th, td {{ padding: 10px 12px; text-align: right; border-bottom: 1px solid #334155; }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ color: #94a3b8; font-weight: 600; background: #0f172a; }}
  code {{ color: #38bdf8; }}
</style>
</head>
<body>
  <div class="header">
    <span class="badge">ULPGC • Ciencia e Ingeniería de Datos</span>
    <span class="badge">IC1/IC2 + ME1/ME2 + Álgebra + OH + AEDV</span>
    <h1>⚡ Cuadro de Mandos AEDV: Cuantización 4-Bit sobre Pesos Reales de <code>SmolLM2-135M</code></h1>
    <p>Análisis exploratorio multidimensional (Curtosis, IQR, Transformación de Yeo-Johnson) y evaluación de <code>Q4_GoldenOpt</code> sobre 4 capas entrenadas reales descargadas de HuggingFace (<code>1,327,104</code> parámetros reales en <code>.safetensors</code>).</p>
  </div>

  <div class="card" style="margin-bottom: 24px;">
    <h3>1. Resumen Estadístico Exploratorio (AEDV) y SQNR en Tensores Reales de SmolLM2-135M</h3>
    <table>
      <thead>
        <tr>
          <th>Tensor Real (SmolLM2-135M)</th>
          <th>Dimensión</th>
          <th>Desv. Típica (σ)</th>
          <th>Rango IQR</th>
          <th>Curtosis Excesiva</th>
          <th>Outliers (&gt;3σ)</th>
          <th>Yeo-Johnson λ*</th>
          <th>Q4_0 (Estándar)</th>
          <th>Q4_Hadamard</th>
          <th>Q4_Stats</th>
          <th>Q4_GoldenOpt (20B)</th>
        </tr>
      </thead>
      <tbody>
        {table_rows_html}
      </tbody>
    </table>
  </div>

  <div class="grid">
    <div class="card"><div id="chart_sqnr" style="height:380px;"></div></div>
    <div class="card"><div id="chart_cascade" style="height:380px;"></div></div>
    <div class="card"><div id="chart_kurt" style="height:360px;"></div></div>
    <div class="card"><div id="chart_modes" style="height:360px;"></div></div>
  </div>

  <script>
    const darkLayout = {{
      paper_bgcolor: '#1e293b', plot_bgcolor: '#0f172a',
      font: {{ color: '#e2e8f0' }},
      margin: {{ t: 48, l: 50, r: 24, b: 60 }}
    }};

    Plotly.newPlot('chart_sqnr', [
      {{ x: [{labels_js}], y: [{q4_0_js}], name: 'Q4_0 (6.40x)', type: 'bar', marker: {{ color: '#64748b' }} }},
      {{ x: [{labels_js}], y: [{q4_had_js}], name: 'Q4_Hadamard (6.40x)', type: 'bar', marker: {{ color: '#a855f7' }} }},
      {{ x: [{labels_js}], y: [{q4_stats_js}], name: 'Q4_Stats (Sparse)', type: 'bar', marker: {{ color: '#f59e0b' }} }},
      {{ x: [{labels_js}], y: [{q4_gold_js}], name: 'Q4_GoldenOpt (6.40x)', type: 'bar', marker: {{ color: '#38bdf8' }} }}
    ], {{ ...darkLayout, title: 'SQNR (dB) por Capa Real de SmolLM2-135M', barmode: 'group', yaxis: {{ title: 'SQNR (dB)' }} }});

    Plotly.newPlot('chart_cascade', [
      {{
        x: ['Q4_0 (6.40x)', 'Q4_Hadamard (6.40x)', 'Q4_Stats (Sparse)', 'Q4_GoldenOpt (6.40x)'],
        y: [{c_t0:.2}, {c_th:.2}, {c_ts:.2}, {c_tg:.2}],
        name: 'Top-1 Token Agreement (%)', type: 'bar',
        marker: {{ color: ['#64748b', '#a855f7', '#f59e0b', '#38bdf8'] }},
        text: ['{c_t0:.2}%', '{c_th:.2}%', '{c_ts:.2}%', '{c_tg:.2}%'], textposition: 'auto'
      }}
    ], {{ ...darkLayout, title: 'Acierto Top-1 Token (%) tras Cascada Real de 4 Capas (SmolLM2-135M)', yaxis: {{ title: 'Top-1 Token Match (%)', range: [0, 100] }} }});

    Plotly.newPlot('chart_kurt', [
      {{ x: [{labels_js}], y: [{kurt_js}], type: 'bar', marker: {{ color: '#f43f5e' }} }}
    ], {{ ...darkLayout, title: 'Curtosis Excesiva de Fisher por Capa (Colas Pesadas Reales)', yaxis: {{ title: 'Curtosis Excesiva (Normal = 0)' }} }});

    Plotly.newPlot('chart_modes', [
      {{
        labels: ['Modo 00 (14 Inliers + ±T1)', 'Modo 01 (16-Level Golden Power-Law)', 'Modo 10 (2-Outlier 6-bit Mantissa)', 'Modo 11 (3-Outlier Cascade)'],
        values: [{m0}, {m1}, {m2}, {m3}],
        type: 'pie', hole: 0.42,
        marker: {{ colors: ['#38bdf8', '#818cf8', '#f59e0b', '#f43f5e'] }}
      }}
    ], {{ ...darkLayout, title: 'Distribución de Modos Hardware Elegidos por Q4_GoldenOpt' }});
  </script>
</body>
</html>"#,
        table_rows_html = table_rows_html,
        labels_js = labels_js,
        q4_0_js = q4_0_js,
        q4_had_js = q4_had_js,
        q4_stats_js = q4_stats_js,
        q4_gold_js = q4_gold_js,
        kurt_js = kurt_js,
        c_t0 = cascade_top1[0],
        c_th = cascade_top1[1],
        c_ts = cascade_top1[2],
        c_tg = cascade_top1[3],
        m0 = total_modes[0],
        m1 = total_modes[1],
        m2 = total_modes[2],
        m3 = total_modes[3],
    );

    let mut f = File::create(output_path.as_ref()).map_err(|e| e.to_string())?;
    f.write_all(html.as_bytes()).map_err(|e| e.to_string())?;
    Ok(())
}
