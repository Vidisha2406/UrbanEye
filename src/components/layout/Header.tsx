import React from 'react';

export const Header: React.FC = () => {
  return (
    <header className="px-8 py-5 border-b border-[#46707E]/15 bg-white/40 backdrop-blur-md sticky top-0 z-20 transition-all">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-2">
        <div>
          <div className="flex items-center space-x-3">
            <h1 className="text-2xl font-extrabold text-[#22393C] tracking-tight">
              From Buses to a Better City
            </h1>
            <span className="hidden sm:inline-flex items-center px-2 py-0.5 rounded-full text-xs font-semibold bg-[#46707E]/10 text-[#46707E] border border-[#46707E]/20">
              Pune Pilot Region
            </span>
          </div>
          <p className="text-xs md:text-sm text-[#46707E] font-medium tracking-wide mt-0.5">
            Mobile Urban Intelligence Platform | AI for Safer Roads, Smarter Transport
          </p>
        </div>
      </div>
    </header>
  );
};
