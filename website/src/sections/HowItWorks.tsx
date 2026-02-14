import { useEffect, useRef, useState, useCallback } from 'react';

// ── Step configuration ──────────────────────────────────────────────
const STEPS = [
  {
    number: '01',
    title: 'Transform',
    body: 'Normalize, label, deduplicate, and synthesize your dataset into a versioned, auditable artifact. Every row is tracked. Every dataset gets a content-addressed version ID and a lineage record.',
    highlight: ['versioned, auditable artifact', 'content-addressed version ID'],
    visualKey: 'transform',
  },
  {
    number: '02',
    title: 'Contract',
    body: 'Each training run must include its config, file hashes, environment snapshot, and eval results. If artifacts are missing or invalid, the pipeline stops before any comparison begins.',
    highlight: ['config, file hashes, environment snapshot', 'pipeline stops'],
    visualKey: 'contract',
  },
  {
    number: '03',
    title: 'Contamination',
    body: 'SHA-256 exact matching and n-gram Jaccard similarity catch both verbatim copies and light paraphrases between train and eval. Configurable thresholds. Zero-tolerance mode for high-stakes evals.',
    highlight: ['verbatim copies and light paraphrases', 'Zero-tolerance mode'],
    visualKey: 'contamination',
  },
  {
    number: '04',
    title: 'Decision',
    body: 'Verifily compares your candidate run against baselines and emits a verdict: SHIP, INVESTIGATE, or DON\'T SHIP — with exit codes your CI already understands.',
    highlight: ['SHIP, INVESTIGATE, or DON\'T SHIP', 'exit codes'],
    visualKey: 'decision',
  },
];

// ── Visual card for each step ───────────────────────────────────────
const StepVisual = ({ activeStep }: { activeStep: number }) => {
  const visuals: Record<string, React.ReactNode> = {
    transform: (
      <div className="bg-slate-50 rounded-2xl p-6 border border-slate-200 shadow-lg">
        <div className="flex items-center gap-3 mb-5">
          <div className="w-10 h-10 rounded-lg bg-blue-500 flex items-center justify-center">
            <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
            </svg>
          </div>
          <span className="text-sm font-semibold text-slate-800">Dataset Pipeline</span>
          <span className="ml-auto text-xs text-slate-400 font-mono">v1 &rarr; v2</span>
        </div>
        {[
          { step: 'Ingest', count: '1,240 rows', status: 'done' },
          { step: 'Normalize', count: '1,240 rows', status: 'done' },
          { step: 'Deduplicate', count: '1,186 rows', status: 'done' },
          { step: 'Label + Synthesize', count: '1,402 rows', status: 'done' },
        ].map((item, i) => (
          <div key={i} className="flex items-center gap-3 py-2.5 border-t border-slate-100">
            <svg className="w-4 h-4 text-emerald-500 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
            </svg>
            <span className="text-sm text-slate-700">{item.step}</span>
            <span className="ml-auto text-xs text-slate-400 font-mono">{item.count}</span>
          </div>
        ))}
        <div className="mt-3 pt-3 border-t border-slate-200">
          <span className="text-xs text-slate-500 font-mono">version_id: a3f8c1d2e4b7</span>
        </div>
      </div>
    ),

    contract: (
      <div className="bg-slate-50 rounded-2xl p-6 border border-slate-200 shadow-lg">
        <div className="flex items-center gap-3 mb-5">
          <div className="w-10 h-10 rounded-lg bg-indigo-500 flex items-center justify-center">
            <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <span className="text-sm font-semibold text-slate-800">Run Contract</span>
          <span className="ml-auto text-xs bg-emerald-100 text-emerald-700 px-2 py-0.5 rounded font-medium">VALID</span>
        </div>
        {[
          { file: 'config.yaml', status: true },
          { file: 'hashes.json', status: true },
          { file: 'environment.json', status: true },
          { file: 'eval/eval_results.json', status: true },
        ].map((item, i) => (
          <div key={i} className="flex items-center gap-3 py-2.5 border-t border-slate-100">
            <svg className={`w-4 h-4 flex-shrink-0 ${item.status ? 'text-emerald-500' : 'text-red-500'}`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
              <path strokeLinecap="round" strokeLinejoin="round" d={item.status ? "M5 13l4 4L19 7" : "M6 18L18 6M6 6l12 12"} />
            </svg>
            <span className="text-sm text-slate-700 font-mono">{item.file}</span>
          </div>
        ))}
        <div className="mt-3 pt-3 border-t border-slate-200">
          <span className="text-xs text-slate-500 font-mono">chain_hash: 9e2f...b41a</span>
        </div>
      </div>
    ),

    contamination: (
      <div className="bg-slate-50 rounded-2xl p-6 border border-slate-200 shadow-lg">
        <div className="flex items-center gap-3 mb-5">
          <div className="w-10 h-10 rounded-lg bg-red-500 flex items-center justify-center">
            <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
          <span className="text-sm font-semibold text-slate-800">Leakage Report</span>
          <span className="ml-auto text-xs bg-red-100 text-red-700 px-2 py-0.5 rounded font-medium">FAIL</span>
        </div>
        <div className="space-y-3">
          <div className="flex justify-between py-2 border-t border-slate-100">
            <span className="text-sm text-slate-600">Exact overlaps</span>
            <span className="text-sm text-red-600 font-semibold font-mono">5 (0.333)</span>
          </div>
          <div className="flex justify-between py-2 border-t border-slate-100">
            <span className="text-sm text-slate-600">Near duplicates</span>
            <span className="text-sm text-amber-600 font-semibold font-mono">7 (0.583)</span>
          </div>
          <div className="flex justify-between py-2 border-t border-slate-100">
            <span className="text-sm text-slate-600">Jaccard threshold</span>
            <span className="text-sm text-slate-500 font-mono">0.8</span>
          </div>
        </div>
        <div className="mt-3 pt-3 border-t border-slate-200 text-xs text-red-600 font-mono">
          Flagged: exact_03, exact_04, exact_06, near_02, near_05...
        </div>
      </div>
    ),

    decision: (
      <div className="bg-slate-50 rounded-2xl p-6 border border-slate-200 shadow-lg">
        <div className="flex items-center gap-3 mb-5">
          <div className="w-10 h-10 rounded-lg bg-slate-800 flex items-center justify-center">
            <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
            </svg>
          </div>
          <span className="text-sm font-semibold text-slate-800">Decision Summary</span>
        </div>
        <div className="bg-red-50 border border-red-200 rounded-xl p-4 mb-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-slate-600">Recommendation</span>
            <span className="text-sm font-bold text-red-600">DON'T SHIP</span>
          </div>
          <p className="text-xs text-red-600">Contamination gate FAIL: dataset leakage detected</p>
        </div>
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-slate-500">f1</span>
            <span className="text-slate-800 font-mono">0.728 (+0.013)</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-slate-500">Reproducibility</span>
            <span className="text-emerald-600 font-mono">verified</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-slate-500">Exit code</span>
            <span className="text-red-600 font-mono font-bold">1</span>
          </div>
        </div>
      </div>
    ),
  };

  return (
    <div className="relative w-full">
      {STEPS.map((step, i) => (
        <div
          key={step.visualKey}
          className="absolute inset-0 transition-all duration-500 ease-out"
          style={{
            opacity: activeStep === i ? 1 : 0,
            transform: `translateY(${activeStep === i ? 0 : 20}px)`,
            pointerEvents: activeStep === i ? 'auto' : 'none',
          }}
        >
          {visuals[step.visualKey]}
        </div>
      ))}
      <div className="invisible">{visuals[STEPS[0].visualKey]}</div>
    </div>
  );
};

// ── Highlighted text renderer ───────────────────────────────────────
const renderHighlightedText = (text: string, highlights: string[]) => {
  let result = text;
  highlights.forEach((h) => {
    result = result.replace(h, `<span class="font-semibold text-slate-900">${h}</span>`);
  });
  return <span dangerouslySetInnerHTML={{ __html: result }} />;
};

// ── Main component ──────────────────────────────────────────────────
const HowItWorks = () => {
  const sectionRef = useRef<HTMLDivElement>(null);
  const pinContainerRef = useRef<HTMLDivElement>(null);
  const [activeStep, setActiveStep] = useState(0);
  const [isMobile, setIsMobile] = useState(false);
  const [prefersReducedMotion, setPrefersReducedMotion] = useState(false);

  useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth < 768);
    checkMobile();
    window.addEventListener('resize', checkMobile);

    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    setPrefersReducedMotion(mq.matches);
    const handler = (e: MediaQueryListEvent) => setPrefersReducedMotion(e.matches);
    mq.addEventListener('change', handler);

    return () => {
      window.removeEventListener('resize', checkMobile);
      mq.removeEventListener('change', handler);
    };
  }, []);

  const setStep = useCallback((progress: number) => {
    const step = Math.min(Math.round(progress * (STEPS.length - 1)), STEPS.length - 1);
    setActiveStep(step);
  }, []);

  useEffect(() => {
    if (!window.gsap || !window.ScrollTrigger) return;
    if (isMobile || prefersReducedMotion) return;

    const gsap = window.gsap;
    const ScrollTrigger = window.ScrollTrigger;

    const timer = setTimeout(() => {
      const ctx = gsap.context(() => {
        ScrollTrigger.create({
          trigger: pinContainerRef.current,
          start: 'top top',
          end: `+=${STEPS.length * window.innerHeight * 0.8}`,
          pin: true,
          scrub: 0.6,
          anticipatePin: 1,
          invalidateOnRefresh: true,
          snap: {
            snapTo: 1 / (STEPS.length - 1),
            duration: { min: 0.15, max: 0.3 },
            delay: 0,
            ease: 'power1.inOut',
          },
          onUpdate: (self: { progress: number }) => {
            setStep(self.progress);
          },
        });
      }, sectionRef);

      (sectionRef.current as any)?.__gsapCtx?.push?.(ctx) || ((sectionRef.current as any).__gsapCtx = ctx);
    }, 150);

    return () => {
      clearTimeout(timer);
      const ctx = (sectionRef.current as any)?.__gsapCtx;
      if (ctx && typeof ctx.revert === 'function') ctx.revert();
    };
  }, [isMobile, prefersReducedMotion, setStep]);

  // Mobile / reduced-motion: stacked layout
  if (isMobile || prefersReducedMotion) {
    return (
      <section ref={sectionRef} className="relative bg-white py-24 overflow-hidden">
        <div className="max-w-5xl mx-auto px-4">
          <h2 className="text-3xl md:text-5xl font-semibold text-slate-900 text-center mb-16">
            How <span className="gradient-text">Verifily</span> works
          </h2>
          <div className="space-y-12">
            {STEPS.map((step) => (
              <div key={step.number} className="grid gap-6">
                <div>
                  <div className="flex items-center gap-3 mb-3">
                    <span className="text-4xl font-bold text-blue-500">{step.number}</span>
                    <h3 className="text-xl font-semibold text-slate-900">{step.title}</h3>
                  </div>
                  <p className="text-lg text-slate-600 leading-relaxed">
                    {renderHighlightedText(step.body, step.highlight)}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    );
  }

  // Desktop: pinned scrollytelling
  return (
    <section ref={sectionRef} className="relative bg-white">
      <div
        ref={pinContainerRef}
        className="relative min-h-screen flex items-center will-change-transform"
      >
        <div className="max-w-6xl mx-auto px-4 w-full">
          <h2 className="text-3xl md:text-5xl font-semibold text-slate-900 text-center mb-16">
            How <span className="gradient-text">Verifily</span> works
          </h2>

          <div className="grid md:grid-cols-2 gap-12 lg:gap-20 items-start">
            {/* Left: step text */}
            <div className="relative">
              <div className="absolute left-0 top-0 bottom-0 w-0.5 bg-slate-200 rounded-full">
                <div
                  className="w-full bg-blue-500 rounded-full transition-all duration-500 ease-out"
                  style={{ height: `${((activeStep + 1) / STEPS.length) * 100}%` }}
                />
              </div>

              <div className="pl-8 space-y-10">
                {STEPS.map((step, i) => {
                  const isActive = activeStep === i;
                  const isPast = activeStep > i;
                  return (
                    <div
                      key={step.number}
                      className={`relative transition-all duration-500 ease-out ${
                        isActive ? 'opacity-100' : isPast ? 'opacity-40' : 'opacity-30'
                      }`}
                    >
                      <div
                        className={`absolute -left-8 top-1 w-4 h-4 rounded-full border-2 transition-all duration-500 ${
                          isActive
                            ? 'bg-blue-500 border-blue-500 scale-125 shadow-lg shadow-blue-500/40'
                            : isPast
                              ? 'bg-blue-500 border-blue-500'
                              : 'bg-white border-slate-300'
                        }`}
                        style={{ transform: `translateX(-6px) ${isActive ? 'scale(1.25)' : ''}` }}
                      />

                      <div className="flex items-center gap-3 mb-2">
                        <span
                          className={`text-5xl font-bold transition-colors duration-500 ${
                            isActive ? 'text-blue-500' : 'text-slate-300'
                          }`}
                        >
                          {step.number}
                        </span>
                        <h3
                          className={`text-xl font-semibold transition-colors duration-500 ${
                            isActive ? 'text-slate-900' : 'text-slate-400'
                          }`}
                        >
                          {step.title}
                        </h3>
                      </div>

                      <p
                        className={`text-lg leading-relaxed transition-colors duration-500 ${
                          isActive ? 'text-slate-600' : 'text-slate-400'
                        }`}
                      >
                        {isActive
                          ? renderHighlightedText(step.body, step.highlight)
                          : step.body}
                      </p>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Right: visual */}
            <div className="sticky top-1/3">
              <StepVisual activeStep={activeStep} />
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HowItWorks;
