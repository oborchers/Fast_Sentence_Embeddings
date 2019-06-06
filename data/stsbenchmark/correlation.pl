#!/usr/bin/perl


=head1 $0

=head1 SYNOPSIS

 correlation.pl gs system

 Outputs the Pearson correlation.

 Example:

   $ ./correlation.pl gs sys 

 Author: Eneko Agirre, Aitor Gonzalez-Agirre

 Dec. 31, 2012

=cut

use Getopt::Long qw(:config auto_help); 
use Pod::Usage; 
use warnings;
use strict;
use Math::Complex;

pod2usage if $#ARGV != 1 ;

if (-e $ARGV[1]) {
    my $continue = 0;
    my %filtered;
    my $do = 0;
    my %a ;
    my %b ;
    my %c ;

    open(I,$ARGV[0]) or die $! ;
    my $filter = 0;
    my $i = 0;
    while (<I>) {
	chomp ;
	next if /^\#/ ;
	if ($_ eq "") {
	    $filter++;
	    $filtered{$filter} = 1;
	}
	else {
	    my @fields = (split(/\t/,$_)) ;
	    my $score = $fields[4] ;
	    warn "wrong range of score in gold standard: $score\n" if ($score > 5) or ($score < 0) ;
	    $a{$i++} = $score ;
	    $filter++;
	}
    } 
    close(I) ;

    my $j = 0 ;

    open(I,$ARGV[1]) or die $! ;
    my $line = 1;
    while (<I>) {
	if(!defined($filtered{$line})) {
	    chomp ;
	    next if /^\#/ ;
	    my @fields = (split(/\s+/,$_)) ; 
	    my ($score) = @fields ;
	    $b{$j} = $score ;
	    $c{$j} = 100;
	    $continue = 1;
	    $j++;
	}
	$line++;
    } 
    close(I) ;
   
    if ($continue == 1) {
	my $sumw=0;

	my $sumwy=0;
	for(my $y = 0; $y < $i; $y++) {
	    $sumwy = $sumwy + (100 * $a{$y});
	    $sumw = $sumw + 100;
	}
	my $meanyw = $sumwy/$sumw;

	my $sumwx=0;
	for(my $x = 0; $x < $i; $x++) {
	    $sumwx = $sumwx + ($c{$x} * $b{$x});
	}
	my $meanxw = $sumwx/$sumw;

	my $sumwxy = 0;
	for(my $x = 0; $x < $i; $x++) {
	    $sumwxy = $sumwxy + $c{$x}*($b{$x} - $meanxw)*($a{$x} - $meanyw);
	}
	my $covxyw = $sumwxy/$sumw;

	my $sumwxx = 0;
	for(my $x = 0; $x < $i; $x++) {
	    $sumwxx = $sumwxx + $c{$x}*($b{$x} - $meanxw)*($b{$x} - $meanxw);
	}
	my $covxxw = $sumwxx/$sumw;

	my $sumwyy = 0;
	for(my $x = 0; $x < $i; $x++) {
	    $sumwyy = $sumwyy + $c{$x}*($a{$x} - $meanyw)*($a{$x} - $meanyw);
	}
	my $covyyw = $sumwyy/$sumw;

	my $corrxyw = $covxyw/sqrt($covxxw*$covyyw);

	printf "Pearson: %.5f\n", $corrxyw ;
    }
}
else{
    printf "Pearson: nan\n";
    exit(1);
}
