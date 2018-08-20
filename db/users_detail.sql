-- phpMyAdmin SQL Dump
-- version 4.6.5.2
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: 20 Agu 2018 pada 06.47
-- Versi Server: 10.1.21-MariaDB
-- PHP Version: 5.6.30

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `movie_rec`
--

-- --------------------------------------------------------

--
-- Struktur dari tabel `users_detail`
--

CREATE TABLE `users_detail` (
  `id` int(4) NOT NULL,
  `name` varchar(50) NOT NULL,
  `email` varchar(30) NOT NULL,
  `password` varchar(30) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

--
-- Dumping data untuk tabel `users_detail`
--

INSERT INTO `users_detail` (`id`, `name`, `email`, `password`) VALUES
(1, 'akunke5', 'admin@gmail.com', 'password'),
(2, 'cobaakun4', 'akun4@gmail.com', 'password'),
(671, 'superadmin', 'admin@gmail.com', 'password'),
(672, 'coba akun 2', 'cobaakun2@gmail.com', 'password'),
(673, 'asdasfadf', 'admin@gmail.com', 'password'),
(674, 'coba ke 6', 'admin@gmail.com', 'password');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `users_detail`
--
ALTER TABLE `users_detail`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `id` (`id`);

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
