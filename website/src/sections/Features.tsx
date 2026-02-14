import { useEffect, useRef } from 'react';
import { ArrowRight } from 'lucide-react';

const Features = () => {
  const sectionRef = useRef<HTMLDivElement>(null);
  const headlineRef = useRef<HTMLDivElement>(null);
  const cardsRef = useRef<HTMLDivElement>(null);
  const bannerRef = useRef<HTMLDivElement>(null);

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

      const cards = cardsRef.current?.querySelectorAll('.feature-card');
      cards?.forEach((card, index) => {
        const directions = [
          { x: -40, y: 50 },
          { x: 40, y: 30 },
          { x: 40, y: 50 },
        ];
        const dir = directions[index] || { x: 0, y: 50 };

        gsap.fromTo(
          card,
          { opacity: 0, x: dir.x, y: dir.y, scale: 0.95 },
          {
            opacity: 1,
            x: 0,
            y: 0,
            scale: 1,
            duration: 0.8,
            delay: index * 0.15,
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
        bannerRef.current,
        { opacity: 0, y: 30 },
        {
          opacity: 1,
          y: 0,
          duration: 0.8,
          delay: 0.4,
          ease: 'power3.out',
          scrollTrigger: {
            trigger: bannerRef.current,
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
      id="features"
      className="relative bg-white py-24 overflow-hidden"
    >
      {/* Headline */}
      <div ref={headlineRef} className="max-w-4xl mx-auto px-4 text-center mb-16">
        <h2 className="text-3xl md:text-5xl font-semibold text-slate-900">
          Core <span className="gradient-text">capabilities</span>
        </h2>
      </div>

      {/* Feature Cards */}
      <div ref={cardsRef} className="max-w-5xl mx-auto px-4">
        <div className="grid md:grid-cols-2 gap-6">
          {/* Dataset Transformation — full height left */}
          <div className="feature-card row-span-2 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-3xl p-8 text-white">
            <h3 className="text-2xl font-bold mb-3">Dataset Transformation</h3>
            <p className="text-white/80 mb-4">
              Clean data in, versioned artifacts out.
            </p>
            <p className="text-white/60 text-sm leading-relaxed">
              Ingest raw data, normalize formats, apply labels, remove duplicates,
              and synthesize training rows — all in a single reproducible pipeline.
              Every dataset gets a content-addressed version ID and a lineage record.
            </p>
            {/* Mini preview */}
            <div className="space-y-2 mt-6">
              {['Ingest', 'Normalize', 'Deduplicate', 'Synthesize'].map((step, i) => (
                <div key={i} className="bg-white/15 backdrop-blur rounded-lg p-3 flex items-center gap-3">
                  <div className="w-8 h-8 rounded-md bg-white/20 flex items-center justify-center text-xs font-bold">
                    {i + 1}
                  </div>
                  <span className="text-white/90 text-sm font-medium">{step}</span>
                  <svg className="ml-auto w-4 h-4 text-white/60" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                  </svg>
                </div>
              ))}
            </div>
          </div>

          {/* Contamination Gate */}
          <div className="feature-card bg-gradient-to-br from-red-500 to-rose-600 rounded-3xl p-8 text-white">
            <h3 className="text-2xl font-bold mb-3">Contamination Gate</h3>
            <p className="text-white/80 mb-2">
              Know when your eval set is lying.
            </p>
            <p className="text-white/60 text-sm leading-relaxed">
              SHA-256 exact matching and n-gram Jaccard similarity catch both
              verbatim copies and light paraphrases. Configurable thresholds.
            </p>
            <div className="mt-6 bg-white/15 backdrop-blur rounded-xl p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-white/70 text-xs">Exact overlaps</span>
                <span className="text-white font-mono text-sm font-bold">5</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-white/70 text-xs">Near duplicates</span>
                <span className="text-white font-mono text-sm font-bold">7</span>
              </div>
            </div>
          </div>

          {/* Decision Artifacts */}
          <div className="feature-card bg-gradient-to-br from-slate-800 to-slate-900 rounded-3xl p-8 text-white">
            <h3 className="text-2xl font-bold mb-3">Decision Artifacts</h3>
            <p className="text-white/80 mb-2">
              A shipping decision, not a dashboard.
            </p>
            <p className="text-white/60 text-sm leading-relaxed">
              Every run produces a decision summary: metrics, deltas against baseline,
              risk flags, and a single recommendation. JSON file + exit code.
            </p>
            <div className="mt-6 bg-white/10 backdrop-blur rounded-xl p-4 font-mono text-xs">
              <div className="text-slate-400">{`{`}</div>
              <div className="text-slate-400 pl-4">"decision": <span className="text-red-400">"DON'T SHIP"</span>,</div>
              <div className="text-slate-400 pl-4">"exit_code": <span className="text-white">1</span>,</div>
              <div className="text-slate-400 pl-4">"blockers": [<span className="text-amber-400">...</span>]</div>
              <div className="text-slate-400">{`}`}</div>
            </div>
          </div>
        </div>
      </div>

      {/* CTA Banner */}
      <div ref={bannerRef} className="max-w-5xl mx-auto px-4 mt-12">
        <div className="relative overflow-hidden rounded-2xl bg-slate-900 cursor-pointer group">
          <div className="absolute inset-0 bg-gradient-to-r from-blue-500/20 via-indigo-500/20 to-violet-500/20 group-hover:opacity-80 transition-opacity" />
          <div className="relative flex items-center justify-between px-8 py-6">
            <p className="text-white text-xl font-semibold">
              The missing step between training and production.
            </p>
            <ArrowRight className="w-6 h-6 text-white group-hover:translate-x-2 transition-transform" />
          </div>
        </div>
      </div>
    </section>
  );
};

export default Features;
