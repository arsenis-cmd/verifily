import { useEffect, useRef } from 'react';

const Stats = () => {
  const sectionRef = useRef<HTMLDivElement>(null);
  const headlineRef = useRef<HTMLDivElement>(null);
  const cardsRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!window.gsap || !window.ScrollTrigger) return;

    const gsap = window.gsap;

    const ctx = gsap.context(() => {
      gsap.fromTo(
        headlineRef.current,
        { opacity: 0, y: 40 },
        {
          opacity: 1,
          y: 0,
          duration: 0.8,
          ease: 'power3.out',
          scrollTrigger: {
            trigger: headlineRef.current,
            start: 'top 80%',
            toggleActions: 'play none none reverse',
          },
        }
      );

      const cards = cardsRef.current?.querySelectorAll('.stat-card');
      cards?.forEach((card, index) => {
        gsap.fromTo(
          card,
          { opacity: 0, y: 60, scale: 0.95 },
          {
            opacity: 1,
            y: 0,
            scale: 1,
            duration: 0.9,
            delay: index * 0.2,
            ease: 'power3.out',
            scrollTrigger: {
              trigger: cardsRef.current,
              start: 'top 75%',
              toggleActions: 'play none none reverse',
            },
          }
        );
      });

      gsap.fromTo(
        bottomRef.current,
        { opacity: 0, y: 20 },
        {
          opacity: 1,
          y: 0,
          duration: 0.5,
          delay: 0.3,
          ease: 'power2.out',
          scrollTrigger: {
            trigger: bottomRef.current,
            start: 'top 85%',
            toggleActions: 'play none none reverse',
          },
        }
      );
    }, sectionRef);

    return () => ctx.revert();
  }, []);

  return (
    <section
      ref={sectionRef}
      id="proof"
      className="relative bg-white py-24 overflow-hidden"
    >
      {/* Headline */}
      <div ref={headlineRef} className="max-w-4xl mx-auto px-4 text-center mb-16">
        <h2 className="text-3xl md:text-5xl font-semibold text-slate-900 mb-4">
          Measured, not claimed.
        </h2>
        <p className="text-slate-500 text-lg max-w-2xl mx-auto">
          Controlled experiments across three training configurations.
          Human-derived synthetic data, transformed and validated through Verifily's pipeline.
        </p>
      </div>

      {/* Stats Cards */}
      <div ref={cardsRef} className="max-w-4xl mx-auto px-4 mb-12">
        <div className="grid md:grid-cols-2 gap-6">
          {/* +1.60 F1 Card */}
          <div className="stat-card bg-gradient-to-br from-blue-500 to-indigo-600 rounded-3xl p-8 text-white">
            <div className="text-7xl md:text-8xl font-bold mb-4">
              +1.60
            </div>
            <p className="text-lg font-medium mb-2">F1 vs AI-contaminated data</p>
            <p className="text-white/70 text-sm leading-relaxed">
              Human-derived synthetic training data outperformed AI-generated training data
              when both were evaluated on a clean, uncontaminated eval set.
            </p>
          </div>

          {/* +0.78 F1 Card */}
          <div className="stat-card bg-gradient-to-br from-violet-500 to-purple-600 rounded-3xl p-8 text-white">
            <div className="text-7xl md:text-8xl font-bold mb-4">
              +0.78
            </div>
            <p className="text-lg font-medium mb-2">F1 vs raw human baseline</p>
            <p className="text-white/70 text-sm leading-relaxed">
              The same synthetic data also surpassed the raw human-only baseline,
              demonstrating that transformation and validation improve training quality.
            </p>
          </div>
        </div>
      </div>

      {/* Bottom line */}
      <div ref={bottomRef} className="max-w-4xl mx-auto px-4">
        <div className="bg-slate-50 rounded-2xl p-6 border border-slate-200 text-center">
          <p className="text-slate-600 font-mono text-sm">
            217 tests &middot; No network &middot; No GPU &middot; Deterministic &middot; Runs in under a second
          </p>
        </div>
      </div>
    </section>
  );
};

export default Stats;
