import { createFileRoute, Link } from "@tanstack/react-router";
import { Activity, BarChart3, Brain, Download, GitCompare, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";

export const Route = createFileRoute("/")({
  component: Landing,
  head: () => ({
    meta: [
      { title: "SpectraML — AI Frequency Extrapolation & Model Benchmarking" },
      { name: "description", content: "Upload scientific datasets, train ML models, extrapolate THz/S-parameter behavior, and export research-grade results." },
    ],
  }),
});

function Landing() {
  return (
    <div className="min-h-screen">
      <nav className="flex items-center justify-between px-8 py-5 border-b border-border/50">
        <div className="flex items-center gap-2">
          <div className="h-8 w-8 rounded-md bg-[var(--gradient-hero)] grid place-items-center">
            <Activity className="h-4 w-4 text-background" />
          </div>
          <span className="font-mono text-sm tracking-tight">SpectraML</span>
        </div>
        <Link to="/app">
          <Button size="sm" variant="secondary">Launch dashboard →</Button>
        </Link>
      </nav>

      <section className="max-w-5xl mx-auto px-8 pt-24 pb-20 text-center">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-border/60 bg-card/40 text-xs font-mono text-muted-foreground mb-8">
          <Sparkles className="h-3 w-3 text-primary" /> Research-grade extrapolation
        </div>
        <h1 className="text-5xl md:text-7xl font-semibold tracking-tight leading-[1.05]">
          Predict the unseen<br />
          <span className="bg-[var(--gradient-hero)] bg-clip-text text-transparent">spectrum</span>.
        </h1>
        <p className="mt-6 text-lg text-muted-foreground max-w-2xl mx-auto">
          Upload S-parameter, THz, or any continuous scientific dataset.
          Benchmark 8 regression models, extrapolate beyond the training range, and export publication-ready graphs.
        </p>
        <div className="mt-10 flex justify-center gap-3">
          <Link to="/app">
            <Button size="lg" className="bg-[var(--gradient-hero)] text-background hover:opacity-90 shadow-[var(--shadow-glow)]">
              Start an experiment
            </Button>
          </Link>
        </div>
      </section>

      <section className="max-w-6xl mx-auto px-8 pb-24 grid md:grid-cols-3 gap-4">
        {[
          { icon: Brain, title: "8 Models, ranked", desc: "Linear, Polynomial-2/3/5, KNN×2, LOESS, Boosted ensemble — auto-ranked by R²." },
          { icon: GitCompare, title: "Split sweeps", desc: "Compare 90/10 → 50/50 splits side-by-side and recommend the winner." },
          { icon: BarChart3, title: "Extrapolation", desc: "Predict any out-of-range window. Curve continuity preserved." },
          { icon: Download, title: "Research export", desc: "CSV predictions, PNG plots, JSON metrics. PDF report on the way." },
          { icon: Activity, title: "Auto preprocessing", desc: "Range detection, missing-value handling, outlier flags." },
          { icon: Sparkles, title: "Validation", desc: "Compare against expected behavior; deviation markers highlight drift." },
        ].map((f) => (
          <div key={f.title} className="rounded-xl border border-border/60 bg-card/40 p-5 hover:border-primary/40 transition-colors">
            <f.icon className="h-5 w-5 text-primary mb-3" />
            <div className="font-medium">{f.title}</div>
            <p className="text-sm text-muted-foreground mt-1">{f.desc}</p>
          </div>
        ))}
      </section>
    </div>
  );
}
