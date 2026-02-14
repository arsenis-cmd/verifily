import { useEffect, useRef, useState } from 'react';
import { ChevronRight, ArrowRight } from 'lucide-react';

// ── Terminal demo showing Verifily pipeline output ──────────────────
const TerminalDemo = () => (
  <div className="relative bg-[#0d1117] rounded-2xl border border-slate-700/50 shadow-2xl overflow-hidden">
    {/* Terminal header */}
    <div className="flex items-center gap-2 px-4 py-3 bg-[#161b22] border-b border-slate-700/30">
      <div className="w-3 h-3 rounded-full bg-[#ff5f57]" />
      <div className="w-3 h-3 rounded-full bg-[#febc2e]" />
      <div className="w-3 h-3 rounded-full bg-[#28c840]" />
      <span className="ml-3 text-xs text-slate-500 font-mono">terminal — verifily</span>
    </div>

    {/* Terminal body */}
    <div className="p-5 md:p-6 font-mono text-[13px] leading-relaxed">
      <div className="text-slate-500">$ verifily pipeline --ci</div>

      <div className="mt-4 space-y-3">
        {/* Contract */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-emerald-400">&#10003;</span>
            <span className="text-slate-300">CONTRACT</span>
          </div>
          <span className="text-emerald-400/70 text-xs">PASS</span>
        </div>
        <div className="text-slate-600 text-xs pl-5">config.yaml, hashes.json, environment.json, eval_results.json</div>

        {/* Contamination */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-red-400">&#10007;</span>
            <span className="text-slate-300">CONTAMINATION</span>
          </div>
          <span className="text-red-400/70 text-xs">FAIL</span>
        </div>
        <div className="text-slate-600 text-xs pl-5">Exact overlaps: 5 (0.333) &middot; Near duplicates: 7 (0.583)</div>

        {/* Regression */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-emerald-400">&#10003;</span>
            <span className="text-slate-300">REGRESSION</span>
          </div>
          <span className="text-emerald-400/70 text-xs">PASS</span>
        </div>
        <div className="text-slate-600 text-xs pl-5">f1: 0.728 (+0.013 vs baseline)</div>

        {/* Divider */}
        <div className="border-t border-slate-700/40 my-1" />

        {/* Decision */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-red-400 font-bold">&#9656;</span>
            <span className="text-white font-medium">DECISION</span>
          </div>
          <span className="text-red-400 font-bold text-xs tracking-wide">DON'T SHIP</span>
        </div>
        <div className="text-slate-600 text-xs pl-5">Blocker: dataset leakage detected between train and eval</div>

        {/* Exit code */}
        <div className="mt-3 pt-2 border-t border-slate-700/40">
          <span className="text-slate-500">$ echo $?</span>
          <div className="text-white mt-1">1</div>
        </div>
      </div>
    </div>
  </div>
);

// ── Main Hero component ─────────────────────────────────────────────
const Hero = () => {
  const sectionRef = useRef<HTMLDivElement>(null);
  const headlineRef = useRef<HTMLDivElement>(null);
  const ctaRef = useRef<HTMLDivElement>(null);
  const demoRef = useRef<HTMLDivElement>(null);
  const glowRef = useRef<HTMLDivElement>(null);
  const [reducedMotion, setReducedMotion] = useState(false);

  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    setReducedMotion(mq.matches);
    const handler = (e: MediaQueryListEvent) => setReducedMotion(e.matches);
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, []);

  useEffect(() => {
    if (!window.gsap || !window.ScrollTrigger || reducedMotion) return;

    const gsap = window.gsap;

    const ctx = gsap.context(() => {
      // Intro fade-in
      const intro = gsap.timeline({ defaults: { ease: 'power3.out' } });
      intro
        .fromTo(
          headlineRef.current?.querySelectorAll('.hero-anim') || [],
          { opacity: 0, y: 40 },
          { opacity: 1, y: 0, duration: 0.9, stagger: 0.1 }
        )
        .fromTo(
          ctaRef.current,
          { opacity: 0, y: 30 },
          { opacity: 1, y: 0, duration: 0.7 },
          '-=0.5'
        );

      // Scroll-driven hero reveal
      const tl = gsap.timeline({
        scrollTrigger: {
          trigger: sectionRef.current,
          start: 'top top',
          end: '+=120%',
          scrub: 0.5,
          pin: true,
          pinSpacing: true,
          anticipatePin: 1,
          invalidateOnRefresh: true,
        },
      });

      tl
        .to(ctaRef.current, { y: -60, opacity: 0, duration: 0.3, ease: 'none' }, 0)
        .to(headlineRef.current, { y: -120, opacity: 0, duration: 0.4, ease: 'none' }, 0.05)
        .to(glowRef.current, { scale: 1.4, opacity: 0.15, duration: 1, ease: 'none' }, 0)
        .fromTo(
          demoRef.current,
          { y: 400, opacity: 0, scale: 0.96 },
          { y: 0, opacity: 1, scale: 1, duration: 0.8, ease: 'none' },
          0.2
        );
    }, sectionRef);

    return () => ctx.revert();
  }, [reducedMotion]);

  // Reduced motion: static layout
  if (reducedMotion) {
    return (
      <section className="relative bg-black overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-transparent via-black/50 to-black pointer-events-none" />
        <div className="relative z-10 py-24 px-4">
          <div className="max-w-5xl mx-auto text-center mb-16">
            <h1 className="text-4xl md:text-6xl lg:text-7xl font-semibold text-white leading-tight mb-6">
              The release gate<br />
              for <span className="gradient-text">machine learning.</span>
            </h1>
            <p className="text-white/70 text-lg md:text-xl max-w-2xl mx-auto mb-8">
              The pipeline between your last training run and production. It checks your data, catches leakage, and tells you whether to ship.
            </p>
            <div className="flex items-center justify-center gap-4">
              <button className="cta-gradient text-white font-medium px-8 py-4 rounded-full inline-flex items-center gap-2">
                Get started <ChevronRight className="w-5 h-5" />
              </button>
              <a href="#" className="text-white/70 hover:text-white font-medium px-6 py-4 inline-flex items-center gap-2 transition-colors">
                Read the docs <ArrowRight className="w-4 h-4" />
              </a>
            </div>
          </div>
          <div className="max-w-3xl mx-auto">
            <TerminalDemo />
          </div>
        </div>
      </section>
    );
  }

  // Animated hero with pinned scroll reveal
  return (
    <section
      ref={sectionRef}
      className="relative h-screen bg-black overflow-hidden"
    >
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-black/50 to-black pointer-events-none" />
      <div
        ref={glowRef}
        className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[400px] blur-3xl pointer-events-none will-change-transform"
        style={{ background: 'radial-gradient(ellipse, rgba(99,102,241,0.15) 0%, transparent 70%)' }}
      />

      {/* Headline + bullets + CTA */}
      <div className="absolute inset-0 flex flex-col items-center justify-center z-10 px-4">
        <div ref={headlineRef} className="text-center max-w-5xl will-change-transform">
          <h1 className="hero-anim text-4xl md:text-6xl lg:text-7xl font-semibold text-white leading-tight mb-6">
            The release gate
            <br />
            for <span className="gradient-text">machine learning.</span>
          </h1>
          <p className="hero-anim text-white/70 text-lg md:text-xl max-w-2xl mx-auto mb-8">
            The pipeline between your last training run and production.
            It checks your data, catches leakage, and tells you whether to ship.
          </p>
          <div className="hero-anim flex flex-col items-start gap-3 max-w-lg mx-auto text-left mb-2">
            <div className="flex items-start gap-3">
              <span className="text-blue-400 mt-0.5 flex-shrink-0">&#10003;</span>
              <span className="text-white/60 text-sm">Detect contamination between train and eval sets before metrics lie to you</span>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-blue-400 mt-0.5 flex-shrink-0">&#10003;</span>
              <span className="text-white/60 text-sm">Enforce reproducible run contracts — config, hashes, environment, results</span>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-blue-400 mt-0.5 flex-shrink-0">&#10003;</span>
              <span className="text-white/60 text-sm">Get a machine-readable decision: SHIP, INVESTIGATE, or DON'T SHIP</span>
            </div>
          </div>
        </div>

        <div ref={ctaRef} className="mt-8 flex items-center gap-4 will-change-transform">
          <button className="cta-gradient text-white font-medium px-8 py-4 rounded-full flex items-center gap-2 hover:opacity-90 transition-opacity">
            Get started
            <ChevronRight className="w-5 h-5" />
          </button>
          <a href="#" className="text-white/70 hover:text-white font-medium px-4 py-4 flex items-center gap-2 transition-colors">
            Read the docs
            <ArrowRight className="w-4 h-4" />
          </a>
        </div>
      </div>

      {/* Terminal demo — revealed on scroll */}
      <div className="absolute inset-0 flex items-center justify-center z-10 pointer-events-none">
        <div
          ref={demoRef}
          className="max-w-3xl w-full px-4 opacity-0 will-change-transform pointer-events-auto"
        >
          <TerminalDemo />
        </div>
      </div>
    </section>
  );
};

export default Hero;
