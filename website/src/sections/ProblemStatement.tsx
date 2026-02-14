import { useEffect, useRef } from 'react';

const ProblemStatement = () => {
  const sectionRef = useRef<HTMLDivElement>(null);
  const textRef = useRef<HTMLDivElement>(null);
  const cardsRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!window.gsap || !window.ScrollTrigger) return;

    const gsap = window.gsap;

    const ctx = gsap.context(() => {
      gsap.fromTo(
        textRef.current,
        { opacity: 0, y: 60 },
        {
          opacity: 1,
          y: 0,
          duration: 1,
          ease: 'power3.out',
          scrollTrigger: {
            trigger: textRef.current,
            start: 'top 75%',
            toggleActions: 'play none none reverse',
          },
        }
      );

      const cards = cardsRef.current?.querySelectorAll('.floating-card');
      cards?.forEach((card, index) => {
        const directions = [
          { x: -100, y: -50 },
          { x: 100, y: -30 },
          { x: -80, y: 50 },
          { x: 60, y: -80 },
          { x: -60, y: 80 },
        ];
        const dir = directions[index % directions.length];

        gsap.fromTo(
          card,
          { opacity: 0, x: dir.x, y: dir.y, scale: 0.8 },
          {
            opacity: 1,
            x: 0,
            y: 0,
            scale: 1,
            duration: 0.9,
            delay: index * 0.12,
            ease: 'back.out(1.4)',
            scrollTrigger: {
              trigger: sectionRef.current,
              start: 'top 60%',
              toggleActions: 'play none none reverse',
            },
          }
        );

        gsap.to(card, {
          y: '+=12',
          duration: 2.5 + index * 0.4,
          repeat: -1,
          yoyo: true,
          ease: 'sine.inOut',
          delay: index * 0.3,
        });
      });

      gsap.fromTo(
        bottomRef.current,
        { opacity: 0, y: 50 },
        {
          opacity: 1,
          y: 0,
          duration: 0.8,
          ease: 'power3.out',
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
      className="relative min-h-screen bg-black py-32 overflow-hidden"
    >
      {/* Floating warning snippets */}
      <div ref={cardsRef} className="absolute inset-0 pointer-events-none">
        <div className="floating-card absolute top-20 left-8 md:left-16">
          <div className="bg-red-950/80 backdrop-blur rounded-xl p-4 border border-red-800/30 w-64 shadow-xl">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-red-400 text-xs font-mono">CONTAMINATION</span>
            </div>
            <p className="text-red-300/80 text-sm font-mono">eval set leaked 5 rows into training data</p>
          </div>
        </div>

        <div className="floating-card absolute top-24 right-8 md:right-16">
          <div className="bg-amber-950/80 backdrop-blur rounded-xl p-4 border border-amber-800/30 w-64 shadow-xl">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-amber-400 text-xs font-mono">REGRESSION</span>
            </div>
            <p className="text-amber-300/80 text-sm font-mono">f1 dropped 0.083 since run_05</p>
          </div>
        </div>

        <div className="floating-card absolute bottom-48 left-12 md:left-24">
          <div className="bg-slate-800/80 backdrop-blur rounded-xl p-4 border border-slate-700/50 w-64 shadow-xl">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-slate-400 text-xs font-mono">CONTRACT</span>
            </div>
            <p className="text-slate-400/80 text-sm font-mono">missing: config.yaml, hashes.json</p>
          </div>
        </div>

        <div className="floating-card absolute top-1/3 left-1/4 hidden md:block">
          <div className="w-14 h-14 rounded-xl bg-slate-800/60 border border-slate-700/30 flex items-center justify-center">
            <span className="text-slate-500 text-2xl">?</span>
          </div>
        </div>

        <div className="floating-card absolute bottom-1/3 right-1/4 hidden md:block">
          <div className="w-14 h-14 rounded-xl bg-slate-800/60 border border-slate-700/30 flex items-center justify-center">
            <span className="text-slate-500 text-2xl">?</span>
          </div>
        </div>
      </div>

      {/* Main text */}
      <div ref={textRef} className="relative z-10 max-w-4xl mx-auto px-4 text-center pt-32">
        <h2 className="text-3xl md:text-5xl lg:text-6xl font-semibold text-white leading-tight">
          Your eval metrics look great.
          <br />
          <span className="text-white/50">But did you check for leakage?</span>
        </h2>
      </div>

      {/* Bottom text */}
      <div ref={bottomRef} className="relative z-10 max-w-2xl mx-auto px-4 mt-24 text-center">
        <p className="text-white/50 text-lg md:text-xl leading-relaxed">
          Most ML teams ship models by checking a spreadsheet, eyeballing metrics,
          and hoping nothing regressed. There is no structured gate between
          "training finished" and "model is in production."
        </p>
      </div>
    </section>
  );
};

export default ProblemStatement;
