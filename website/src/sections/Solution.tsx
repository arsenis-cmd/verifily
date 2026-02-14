import { useEffect, useRef } from 'react';
import { ChevronRight } from 'lucide-react';

const Solution = () => {
  const sectionRef = useRef<HTMLDivElement>(null);
  const logoRef = useRef<HTMLDivElement>(null);
  const textRef = useRef<HTMLDivElement>(null);
  const timelineRef = useRef<HTMLDivElement>(null);
  const bottomTextRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!window.gsap || !window.ScrollTrigger) return;

    const gsap = window.gsap;

    const ctx = gsap.context(() => {
      gsap.fromTo(
        logoRef.current,
        { scale: 0, opacity: 0, rotation: -180 },
        {
          scale: 1,
          opacity: 1,
          rotation: 0,
          duration: 1,
          ease: 'back.out(1.7)',
          scrollTrigger: {
            trigger: logoRef.current,
            start: 'top 75%',
            toggleActions: 'play none none reverse',
          },
        }
      );

      gsap.fromTo(
        textRef.current,
        { opacity: 0, y: 50 },
        {
          opacity: 1,
          y: 0,
          duration: 0.9,
          ease: 'power3.out',
          scrollTrigger: {
            trigger: textRef.current,
            start: 'top 75%',
            toggleActions: 'play none none reverse',
          },
        }
      );

      const timelineItems = timelineRef.current?.querySelectorAll('.timeline-item');
      timelineItems?.forEach((item, index) => {
        gsap.fromTo(
          item,
          { opacity: 0, y: 40, scale: 0.9 },
          {
            opacity: 1,
            y: 0,
            scale: 1,
            duration: 0.7,
            delay: index * 0.15,
            ease: 'power3.out',
            scrollTrigger: {
              trigger: timelineRef.current,
              start: 'top 70%',
              toggleActions: 'play none none reverse',
            },
          }
        );
      });

      const progressLine = timelineRef.current?.querySelector('.progress-line');
      if (progressLine) {
        gsap.fromTo(
          progressLine,
          { scaleX: 0 },
          {
            scaleX: 1,
            duration: 1.5,
            ease: 'power2.out',
            scrollTrigger: {
              trigger: timelineRef.current,
              start: 'top 65%',
              toggleActions: 'play none none reverse',
            },
          }
        );
      }

      gsap.fromTo(
        bottomTextRef.current,
        { opacity: 0, y: 40 },
        {
          opacity: 1,
          y: 0,
          duration: 0.8,
          ease: 'power3.out',
          scrollTrigger: {
            trigger: bottomTextRef.current,
            start: 'top 80%',
            toggleActions: 'play none none reverse',
          },
        }
      );
    }, sectionRef);

    return () => ctx.revert();
  }, []);

  const pipelineSteps = [
    {
      id: 1,
      name: 'Transform',
      desc: 'Normalize, label, dedupe',
      icon: 'data',
      completed: true,
    },
    {
      id: 2,
      name: 'Contract',
      desc: 'Validate run artifacts',
      icon: 'check',
      completed: true,
    },
    {
      id: 3,
      name: 'Verifily',
      desc: '',
      icon: 'logo',
      isCenter: true,
      completed: false,
    },
    {
      id: 4,
      name: 'Contamination',
      desc: 'Detect train/eval overlap',
      icon: 'shield',
      completed: false,
    },
    {
      id: 5,
      name: 'Decision',
      desc: 'SHIP / DON\'T SHIP',
      icon: 'flag',
      completed: false,
    },
  ];

  return (
    <section
      ref={sectionRef}
      id="how-it-works"
      className="relative min-h-screen bg-white py-32 overflow-hidden"
    >
      {/* Logo */}
      <div ref={logoRef} className="flex justify-center mb-12">
        <div className="w-24 h-24 rounded-2xl logo-gradient flex items-center justify-center shadow-xl">
          <svg viewBox="0 0 24 24" className="w-12 h-12 text-white" fill="currentColor">
            <path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4zm-2 16l-4-4 1.41-1.41L10 14.17l6.59-6.59L18 9l-8 8z"/>
          </svg>
        </div>
      </div>

      {/* Headline */}
      <div ref={textRef} className="max-w-3xl mx-auto px-4 text-center mb-20">
        <h2 className="text-3xl md:text-5xl font-semibold text-slate-900 leading-tight">
          Four checks.
          <br />
          <span className="gradient-text">One decision.</span>
        </h2>
      </div>

      {/* Pipeline timeline */}
      <div ref={timelineRef} className="max-w-5xl mx-auto px-4 mb-20">
        <div className="relative">
          {/* Progress line */}
          <div className="absolute top-8 left-0 right-0 h-0.5 bg-slate-200">
            <div
              className="progress-line h-full bg-gradient-to-r from-blue-500 via-indigo-500 to-violet-500 origin-left"
              style={{ width: '40%' }}
            />
          </div>

          {/* Steps */}
          <div className="relative flex justify-between items-start">
            {pipelineSteps.map((step) => (
              <div key={step.id} className="timeline-item relative flex flex-col items-center">
                {step.isCenter ? (
                  <div className="relative z-10">
                    <div className="w-20 h-20 rounded-2xl logo-gradient flex items-center justify-center shadow-lg">
                      <svg viewBox="0 0 24 24" className="w-10 h-10 text-white" fill="currentColor">
                        <path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4zm-2 16l-4-4 1.41-1.41L10 14.17l6.59-6.59L18 9l-8 8z"/>
                      </svg>
                    </div>
                  </div>
                ) : (
                  <div className="flex flex-col items-center">
                    <div
                      className={`w-16 h-16 rounded-xl flex items-center justify-center mb-4 z-10 transition-all duration-500 ${
                        step.completed
                          ? 'bg-blue-500 shadow-lg shadow-blue-500/30'
                          : 'bg-white border-4 border-slate-200'
                      }`}
                    >
                      {step.completed ? (
                        <svg className="w-7 h-7 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                        </svg>
                      ) : (
                        <div className="w-3 h-3 rounded-full bg-slate-300" />
                      )}
                    </div>

                    <div className="bg-white rounded-xl p-4 shadow-lg border border-slate-200 w-36 text-center">
                      <p className="text-sm font-semibold text-slate-800">{step.name}</p>
                      <p className="text-xs text-slate-500 mt-1">{step.desc}</p>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Bottom text */}
      <div ref={bottomTextRef} className="max-w-3xl mx-auto px-4 text-center mb-10">
        <h3 className="text-2xl md:text-3xl font-semibold text-slate-900 leading-tight mb-4">
          One command. Machine-readable output.
        </h3>
        <p className="text-slate-500 text-lg">
          Runs in CI. Reads artifacts from disk. Exit code 0 means ship.
        </p>
      </div>

      {/* CTA */}
      <div className="flex justify-center">
        <button className="bg-slate-900 text-white font-medium px-8 py-4 rounded-full flex items-center gap-2 hover:bg-slate-800 transition-colors shadow-lg hover:shadow-xl">
          Get started
          <ChevronRight className="w-5 h-5" />
        </button>
      </div>
    </section>
  );
};

export default Solution;
