"use client";

import { useMemo } from "react";
import anchorsData from "./landing_grid_anchors.json";

const ANCHOR_METADATA = {
  "1sdtlwi": {
    title: "Neon Surrealism & Tech-Horror",
    description: "Vibrant, unsettling, and psychological. Think neon lights, cyber-realities, and mind-bending thrillers (Videodrome, Perfect Blue).",
    imageSrc: "/images/vibe_neon_surrealism.png"
  },
  "1pkbf6n": {
    title: "Cozy & Cold Winter Atmospheres",
    description: "Snowy landscapes, quiet isolation, and dark winter mysteries (Fargo, The Hateful Eight, Sleepy Hollow).",
    imageSrc: "/images/vibe_winter_atmospheres.png"
  },
  "1r5b2t0": {
    title: "Futuristic Action & High-Tech Arena",
    description: "Giant robots, high-stakes sci-fi sports, and adrenaline-fueled battles (Pacific Rim, Speed Racer).",
    imageSrc: "/images/vibe_futuristic_action.png"
  },
  "1rz60b2": {
    title: "Gritty Fantasy & Post-Apocalyptic",
    description: "Rough-hewn adventures, apocalyptic journeys, and dark fantasy (Army of Darkness, The Book of Eli).",
    imageSrc: "/images/vibe_gritty_fantasy.png"
  },
  "1s7emt0": {
    title: "Grungy Youth Culture & Indie Roadtrips",
    description: "Raw, character-driven stories of teenage rebellion, sprawling road trips, and nostalgic party scenes (Alpha Dog, American Honey).",
    imageSrc: "/images/vibe_grungy_youth.png"
  },
  "1szk2bl": {
    title: "Hyper-Stylized Asian Cinema & Romance",
    description: "Melancholic romances, fast-paced thrillers, and neon-drenched urban aesthetics (Chungking Express, 2046, Bullet Train).",
    imageSrc: "/images/vibe_asian_cinema.png"
  }
};

const AnchorCard = ({ title, description, imageSrc, onClick, isSelected }) => (
    <div
        onClick={onClick}
        className={`group flex flex-col rounded-lg overflow-hidden cursor-pointer transition-all duration-300 border-2 ${
            isSelected 
            ? 'bg-purple-800/40 border-purple-500 shadow-lg shadow-purple-500/30 scale-[1.02]' 
            : 'bg-gray-800 border-gray-700 hover:border-purple-500/50 hover:bg-gray-700/50 hover:shadow-md'
        }`}
    >
        {imageSrc ? (
            <div className="relative w-full h-44 overflow-hidden bg-gray-900 border-b border-gray-700">
                <img 
                    src={imageSrc} 
                    alt={title} 
                    className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105" 
                />
                <div className="absolute inset-0 bg-gradient-to-t from-gray-900 via-transparent to-transparent opacity-80" />
            </div>
        ) : (
            <div className="w-full h-44 bg-gray-900 flex items-center justify-center border-b border-gray-700">
                <span className="text-gray-600 text-sm">No Image</span>
            </div>
        )}
        <div className="p-5 flex-1 flex flex-col justify-between">
            <div>
                <h3 className="text-lg font-bold text-white group-hover:text-purple-300 transition-colors duration-300">{title}</h3>
                <p className="text-gray-400 mt-2 text-sm leading-relaxed">{description}</p>
            </div>
        </div>
    </div>
);

export const VibeAnchors = ({ selectedAnchor, onSelectAnchor }) => {
    const anchors = useMemo(() => {
        return anchorsData.map((anchor) => {
            const meta = ANCHOR_METADATA[anchor.post_id] || {
                title: anchor.descriptors.join(", ") || `Vibe Cluster ${anchor.cluster_id}`,
                description: `Featuring: ${anchor.verified_movies.slice(0, 3).join(", ")}`,
                imageSrc: null
            };
            return {
                id: anchor.post_id,
                title: meta.title,
                description: meta.description,
                imageSrc: meta.imageSrc,
                vector: anchor.combined_vector
            };
        });
    }, []);

    // Function to compare arrays by value
    const areVectorsEqual = (vec1, vec2) => {
        if (!vec1 || !vec2 || vec1.length !== vec2.length) return false;
        return vec1.every((value, index) => value === vec2[index]);
    };

    return (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {anchors.map((anchor) => (
                <AnchorCard
                    key={anchor.id}
                    title={anchor.title}
                    description={anchor.description}
                    imageSrc={anchor.imageSrc}
                    onClick={() => onSelectAnchor(anchor.vector)}
                    isSelected={areVectorsEqual(selectedAnchor, anchor.vector)}
                />
            ))}
        </div>
    );
};

